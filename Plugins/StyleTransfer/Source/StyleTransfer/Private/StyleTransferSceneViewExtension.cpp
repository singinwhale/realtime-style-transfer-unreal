#include "StyleTransferSceneViewExtension.h"

#include "CoreGlobals.h"
#include "SceneView.h"

#include "GlobalShader.h"
#include "PipelineStateCache.h"
#include "RenderTargetPool.h"
#include "Shader.h"

#include "RHI.h"
#include "SceneView.h"
#include "ScreenPass.h"
#include "CommonRenderResources.h"
#include "NeuralNetwork.h"
#include "RenderGraphEvent.h"
#include "PostProcess/PostProcessing.h"
#include "Containers/DynamicRHIResourceArray.h"
#include "PostProcess/PostProcessMaterial.h"
#include "OutputTensorToSceneColorCS.h"
#include "SceneColorToInputTensorCS.h"

template <class OutType, class InType>
OutType CastNarrowingSafe(InType InValue)
{
	if (InValue > TNumericLimits<OutType>::Max())
	{
		return TNumericLimits<OutType>::Max();
	}
	if (InValue < TNumericLimits<OutType>::Min())
	{
		return TNumericLimits<OutType>::Min();
	}
	return static_cast<OutType>(InValue);
}


FStyleTransferSceneViewExtension::FStyleTransferSceneViewExtension(const FAutoRegister& AutoRegister, FViewportClient* AssociatedViewportClient, UNeuralNetwork* InStyleTransferNetwork, int32 InInferenceContext)
	: FSceneViewExtensionBase(AutoRegister)
	  , StyleTransferNetwork(InStyleTransferNetwork)
	  , LinkedViewportClient(AssociatedViewportClient)
	  , InferenceContext(InInferenceContext)
{
	ensure(InStyleTransferNetwork->GetDeviceType() == ENeuralDeviceType::GPU);
}

void FStyleTransferSceneViewExtension::SetupViewFamily(FSceneViewFamily& InViewFamily)
{
}

void FStyleTransferSceneViewExtension::SubscribeToPostProcessingPass(EPostProcessingPass PassId,
                                                                     FAfterPassCallbackDelegateArray&
                                                                     InOutPassCallbacks, bool bIsPassEnabled)
{
	if (PassId == EPostProcessingPass::Tonemap)
	{
		InOutPassCallbacks.Add(
			FAfterPassCallbackDelegate::CreateRaw(
				this, &FStyleTransferSceneViewExtension::PostProcessPassAfterTonemap_RenderThread));
	}
}

FScreenPassTexture FStyleTransferSceneViewExtension::PostProcessPassAfterTonemap_RenderThread(
	FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& InOutInputs)
{
	const FSceneViewFamily& ViewFamily = *View.Family;

	const FScreenPassTexture& SceneColor = InOutInputs.Textures[(uint32)EPostProcessMaterialInput::SceneColor];

	if (!EnumHasAnyFlags(SceneColor.Texture->Desc.Flags, TexCreate_ShaderResource))
	{
		return SceneColor;
	}

	if (!SceneColor.IsValid())
	{
		return SceneColor;
	}

	RDG_EVENT_SCOPE(GraphBuilder, "StyleTransfer");

	//Get input and output viewports. Backbuffer could be targeting a different region than input viewport
	const FScreenPassTextureViewport SceneColorViewport(SceneColor);

	FScreenPassRenderTarget SceneColorRenderTarget(SceneColor, ERenderTargetLoadAction::ELoad);

	checkSlow(View.bIsViewInfo);
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);

	/*AddDrawScreenPass(GraphBuilder, RDG_EVENT_NAME("ProcessOCIOColorSpaceXfrm"), ViewInfo, BackBufferViewport,
	                  SceneColorViewport, OCIOPixelShader, Parameters);*/

	const FNeuralTensor& StyleTransferContentInputTensor = StyleTransferNetwork->GetInputTensorForContext(InferenceContext, 0);

	const FIntVector InputTensorDimensions = {
		CastNarrowingSafe<int32>(StyleTransferContentInputTensor.GetSize(1)),
		CastNarrowingSafe<int32>(StyleTransferContentInputTensor.GetSize(2)),
		CastNarrowingSafe<int32>(StyleTransferContentInputTensor.GetSize(3)),
	};
	const FIntPoint SceneColorRenderTargetDimensions = SceneColorRenderTarget.Texture->Desc.Extent;

	FRDGBufferRef StyleTransferContentInputBuffer = GraphBuilder.RegisterExternalBuffer(StyleTransferContentInputTensor.GetPooledBuffer());
	auto SceneColorToInputTensorParameters = GraphBuilder.AllocParameters<FSceneColorToInputTensorCS::FParameters>();
	SceneColorToInputTensorParameters->TensorVolume = CastNarrowingSafe<uint32>(StyleTransferContentInputTensor.Num());
	SceneColorToInputTensorParameters->InputTexture = SceneColorRenderTarget.Texture;
	SceneColorToInputTensorParameters->InputTextureSampler = TStaticSamplerState<SF_Bilinear>::GetRHI();
	SceneColorToInputTensorParameters->OutputUAV = GraphBuilder.CreateUAV(StyleTransferContentInputBuffer);
	SceneColorToInputTensorParameters->OutputDimensions = {InputTensorDimensions.X, InputTensorDimensions.Y};
	SceneColorToInputTensorParameters->HalfPixelUV = FVector2f(0.5f / SceneColorRenderTargetDimensions.X, 0.5 / SceneColorRenderTargetDimensions.Y);
	FIntVector SceneColorToInputTensorGroupCount = FComputeShaderUtils::GetGroupCount(
		{InputTensorDimensions.X, InputTensorDimensions.Y, 1},
		FSceneColorToInputTensorCS::ThreadGroupSize
	);

	TShaderMapRef<FSceneColorToInputTensorCS> SceneColorToInputTensorCS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	GraphBuilder.AddPass(
		RDG_EVENT_NAME("SceneColorToInputTensor"),
		SceneColorToInputTensorParameters,
		ERDGPassFlags::Compute,
		[SceneColorToInputTensorCS, SceneColorToInputTensorParameters, SceneColorToInputTensorGroupCount](FRHICommandList& RHICommandList)
		{
			FComputeShaderUtils::Dispatch(RHICommandList, SceneColorToInputTensorCS,
			                              *SceneColorToInputTensorParameters, SceneColorToInputTensorGroupCount);
		}
	);


	const FNeuralTensor& StyleTransferOutputTensor = StyleTransferNetwork->GetOutputTensorForContext(InferenceContext, 0);
	FIntVector OutputTensorDimensions = {
		CastNarrowingSafe<int32>(StyleTransferOutputTensor.GetSize(1)),
		CastNarrowingSafe<int32>(StyleTransferOutputTensor.GetSize(2)),
		CastNarrowingSafe<int32>(StyleTransferOutputTensor.GetSize(3)),
	};
	// Reusing the same output description for our back buffer as SceneColor
	FRDGTextureDesc OutputDesc = SceneColor.Texture->Desc;
	// this is flipped because the Output tensor has the vertical dimension first
	// while unreal has the horizontal dimension first
	OutputDesc.Extent = {OutputTensorDimensions[1], OutputTensorDimensions[0]};
	OutputDesc.Flags |= TexCreate_RenderTargetable | TexCreate_UAV;
	FLinearColor ClearColor(0., 0., 0., 0.);
	OutputDesc.ClearValue = FClearValueBinding(ClearColor);
	FRDGTexture* StyleTransferRenderTargetTexture = GraphBuilder.CreateTexture(
		OutputDesc, TEXT("StyleTransferRenderTargetTexture"));
	TSharedPtr<FScreenPassRenderTarget> StyleTransferOutputTarget = MakeShared<FScreenPassRenderTarget>(StyleTransferRenderTargetTexture, SceneColor.ViewRect,
	                                                                                                    ERenderTargetLoadAction::EClear);

	StyleTransferNetwork->Run(GraphBuilder, InferenceContext);

	FRDGBufferRef StyleTransferOutputBuffer = GraphBuilder.RegisterExternalBuffer(StyleTransferContentInputTensor.GetPooledBuffer());

	auto OutputTensorToSceneColorParameters = GraphBuilder.AllocParameters<FOutputTensorToSceneColorCS::FParameters>();
	OutputTensorToSceneColorParameters->InputTensor = GraphBuilder.CreateSRV(StyleTransferOutputBuffer, EPixelFormat::PF_FloatRGB);
	OutputTensorToSceneColorParameters->OutputTexture = GraphBuilder.CreateUAV(StyleTransferRenderTargetTexture);
	FIntVector OutputTensorToSceneColorGroupCount = FComputeShaderUtils::GetGroupCount(
		{OutputTensorDimensions.X, OutputTensorDimensions.Y, 1},
		FOutputTensorToSceneColorCS::ThreadGroupSize
	);

	TShaderMapRef<FOutputTensorToSceneColorCS> OutputTensorToSceneColorCS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	GraphBuilder.AddPass(
		RDG_EVENT_NAME("OutputTensorToSceneColor"),
		OutputTensorToSceneColorParameters,
		ERDGPassFlags::Compute,
		[OutputTensorToSceneColorCS, OutputTensorToSceneColorParameters, OutputTensorToSceneColorGroupCount](FRHICommandList& RHICommandList)
		{
			FComputeShaderUtils::Dispatch(RHICommandList, OutputTensorToSceneColorCS,
			                              *OutputTensorToSceneColorParameters, OutputTensorToSceneColorGroupCount);
		}
	);


	TSharedPtr<FScreenPassRenderTarget> BackBufferRenderTarget;
	// If the override output is provided it means that this is the last pass in post processing.
	if (InOutInputs.OverrideOutput.IsValid())
	{
		BackBufferRenderTarget = MakeShared<FScreenPassRenderTarget>(InOutInputs.OverrideOutput);
		// @todo: do not use copy. Resample the styled 1920x960 texture to the fullscreen texture by drawing into the texture
		AddCopyTexturePass(GraphBuilder, StyleTransferRenderTargetTexture, BackBufferRenderTarget->Texture);
	}
	else
	{
		BackBufferRenderTarget = StyleTransferOutputTarget;
	}

	return MoveTemp(*BackBufferRenderTarget);
}
