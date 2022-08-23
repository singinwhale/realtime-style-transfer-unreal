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


FStyleTransferSceneViewExtension::FStyleTransferSceneViewExtension(const FAutoRegister& AutoRegister, FViewportClient* AssociatedViewportClient, UNeuralNetwork* InStyleTransferNetwork)
	: FSceneViewExtensionBase(AutoRegister)
	  , StyleTransferNetwork(InStyleTransferNetwork)
	  , LinkedViewportClient(AssociatedViewportClient)
{
	ensure(InStyleTransferNetwork->GetDeviceType() == ENeuralDeviceType::GPU);
}

void FStyleTransferSceneViewExtension::SetupViewFamily(FSceneViewFamily& InViewFamily)
{
	InferenceContext = StyleTransferNetwork->CreateInferenceContext();
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

	FScreenPassRenderTarget BackBufferRenderTarget;

	// If the override output is provided it means that this is the last pass in post processing.
	if (InOutInputs.OverrideOutput.IsValid())
	{
		BackBufferRenderTarget = InOutInputs.OverrideOutput;
	}
	else
	{
		// Reusing the same output description for our back buffer as SceneColor when it's not overriden
		FRDGTextureDesc OutputDesc = SceneColor.Texture->Desc;
		OutputDesc.Flags |= TexCreate_RenderTargetable;
		FLinearColor ClearColor(0., 0., 0., 0.);
		OutputDesc.ClearValue = FClearValueBinding(ClearColor);

		FRDGTexture* BackBufferRenderTargetTexture = GraphBuilder.CreateTexture(
			OutputDesc, TEXT("BackBufferRenderTargetTexture"));
		BackBufferRenderTarget = FScreenPassRenderTarget(BackBufferRenderTargetTexture, SceneColor.ViewRect,
		                                                 ERenderTargetLoadAction::EClear);
	}

	//Get input and output viewports. Backbuffer could be targeting a different region than input viewport
	const FScreenPassTextureViewport SceneColorViewport(SceneColor);
	const FScreenPassTextureViewport BackBufferViewport(BackBufferRenderTarget);

	FScreenPassRenderTarget SceneColorRenderTarget(SceneColor, ERenderTargetLoadAction::ELoad);

	checkSlow(View.bIsViewInfo);
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);

	/*AddDrawScreenPass(GraphBuilder, RDG_EVENT_NAME("ProcessOCIOColorSpaceXfrm"), ViewInfo, BackBufferViewport,
	                  SceneColorViewport, OCIOPixelShader, Parameters);*/

	auto SceneColorToInputTensorParameters = GraphBuilder.AllocParameters<FSceneColorToInputTensorCS::FParameters>();
	FNeuralTensor InputTensor = StyleTransferNetwork->GetInputTensor();
	SceneColorToInputTensorParameters->TensorVolume = CastNarrowingSafe<uint32>(InputTensor.Num());
	SceneColorToInputTensorParameters->InputTexture = SceneColorRenderTarget.Texture;
	SceneColorToInputTensorParameters->OutputUAV = InputTensor.GetBufferUAVRef();
	FIntVector SceneColorToInputTensorGroupCount;

	TShaderMapRef<FSceneColorToInputTensorCS> SceneColorToInputTensorCS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	ClearUnusedGraphResources(SceneColorToInputTensorCS, SceneColorToInputTensorParameters);
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

	StyleTransferNetwork->GetInputDataPointerMutableForContext(InferenceContext, 0);
	StyleTransferNetwork->Run(GraphBuilder, InferenceContext);

	auto OutputTensorToSceneColorParameters = GraphBuilder.AllocParameters<FOutputTensorToSceneColorCS::FParameters>();
	FNeuralTensor OutputTensor = StyleTransferNetwork->GetOutputTensor(0);
	OutputTensorToSceneColorParameters->InputTensor = OutputTensor.GetBufferSRVRef();
	OutputTensorToSceneColorParameters->OutputTexture = GraphBuilder.CreateUAV(SceneColorRenderTarget.Texture);
	FIntVector OutputTensorToSceneColorGroupCount;

	TShaderMapRef<FOutputTensorToSceneColorCS> OutputTensorToSceneColorCS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	ClearUnusedGraphResources(OutputTensorToSceneColorCS, OutputTensorToSceneColorParameters);
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

	return MoveTemp(BackBufferRenderTarget);
}
