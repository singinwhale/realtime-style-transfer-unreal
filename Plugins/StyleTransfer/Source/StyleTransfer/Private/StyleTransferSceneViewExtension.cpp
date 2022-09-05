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
#include "IPixWinPlugin.h"
#include "IPixWinPlugin.h"
#include "IRenderCaptureProvider.h"
#include "NeuralNetwork.h"
#include "RenderGraphEvent.h"
#include "PostProcess/PostProcessing.h"
#include "Containers/DynamicRHIResourceArray.h"
#include "PostProcess/PostProcessMaterial.h"
#include "OutputTensorToSceneColorCS.h"
#include "PixelShaderUtils.h"
#include "SceneColorToInputTensorCS.h"
#include "StyleTransferModule.h"

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


FStyleTransferSceneViewExtension::FStyleTransferSceneViewExtension(const FAutoRegister& AutoRegister, FViewportClient* AssociatedViewportClient, UNeuralNetwork* InStyleTransferNetwork, TSharedRef<int32> InInferenceContext)
	: FSceneViewExtensionBase(AutoRegister)
	  , StyleTransferNetworkWeakPtr(InStyleTransferNetwork)
	  , StyleTransferNetwork(InStyleTransferNetwork)
	  , LinkedViewportClient(AssociatedViewportClient)
	  , InferenceContext(InInferenceContext)
{
	ensure(InStyleTransferNetwork->GetDeviceType() == ENeuralDeviceType::GPU);
}

void FStyleTransferSceneViewExtension::SetupViewFamily(FSceneViewFamily& InViewFamily)
{
}

bool FStyleTransferSceneViewExtension::IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const
{
	check(IsInGameThread());
	return bIsEnabled && *InferenceContext != -1 && StyleTransferNetworkWeakPtr.IsValid();
}

void FStyleTransferSceneViewExtension::AddRescalingTextureCopy(FRDGBuilder& GraphBuilder, FRDGTexture& RDGSourceTexture, FScreenPassRenderTarget& DestinationRenderTarget)
{
	FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GMaxRHIFeatureLevel);

	TShaderMapRef<FScreenPassVS> VertexShader(ShaderMap);

	TShaderMapRef<FCopyRectPS> PixelShader(ShaderMap);

	FCopyRectPS::FParameters* PixelShaderParameters = GraphBuilder.AllocParameters<FCopyRectPS::FParameters>();
	PixelShaderParameters->InputTexture = &RDGSourceTexture;
	PixelShaderParameters->InputSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PixelShaderParameters->RenderTargets[0] = DestinationRenderTarget.GetRenderTargetBinding();

	ClearUnusedGraphResources(PixelShader, PixelShaderParameters);

	FRHIBlendState* BlendState = FScreenPassPipelineState::FDefaultBlendState::GetRHI();
	FRHIDepthStencilState* DepthStencilState = FScreenPassPipelineState::FDefaultDepthStencilState::GetRHI();

	const FScreenPassPipelineState PipelineState(VertexShader, PixelShader, BlendState, DepthStencilState);

	GraphBuilder.AddPass(
		RDG_EVENT_NAME("RescalingTextureCopy"),
		PixelShaderParameters,
		ERDGPassFlags::Raster,
		[PipelineState, Extent = DestinationRenderTarget.Texture->Desc.Extent, PixelShader, PixelShaderParameters](FRHICommandList& RHICmdList)
		{
			PipelineState.Validate();
			RHICmdList.SetViewport(0.0f, 0.0f, 0.0f, Extent.X, Extent.Y, 1.0f);
			SetScreenPassPipelineState(RHICmdList, PipelineState);
			SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), *PixelShaderParameters);
			DrawRectangle(
				RHICmdList,
				0, 0, Extent.X, Extent.Y,
				0, 0, Extent.X, Extent.Y,
				Extent,
				Extent,
				PipelineState.VertexShader,
				EDRF_UseTriangleOptimization);
		});
}

void FStyleTransferSceneViewExtension::SubscribeToPostProcessingPass(EPostProcessingPass PassId, FAfterPassCallbackDelegateArray& InOutPassCallbacks, bool bIsPassEnabled)
{
	if (PassId == EPostProcessingPass::Tonemap)
	{
		InOutPassCallbacks.Add(
			FAfterPassCallbackDelegate::CreateRaw(
				this, &FStyleTransferSceneViewExtension::PostProcessPassAfterTonemap_RenderThread));
	}
}

void FStyleTransferSceneViewExtension::PreRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily)
{
	return;
	const FName RenderCaptureProviderType = IRenderCaptureProvider::GetModularFeatureName();
	if(!IModularFeatures::Get().IsModularFeatureAvailable(RenderCaptureProviderType))
		return;

	IRenderCaptureProvider& RenderCaptureProvider = IModularFeatures::Get().GetModularFeature<IRenderCaptureProvider>(RenderCaptureProviderType);
	if(bIsEnabled && NumFramesCaptured == -1)
	{
		RenderCaptureProvider.BeginCapture(&GRHICommandList.GetImmediateCommandList());
		NumFramesCaptured = 0;
	}

	if(NumFramesCaptured >= 0)
	{
		++NumFramesCaptured;
	}

	if(NumFramesCaptured == 10)
	{
		RenderCaptureProvider.EndCapture(&GRHICommandList.GetImmediateCommandList());
	}
}

FRDGTexture* FStyleTransferSceneViewExtension::TensorToTexture(FRDGBuilder& GraphBuilder, const FRDGTextureDesc& BaseDestinationDesc, const FNeuralTensor& SourceTensor)
{
	FIntVector SourceTensorDimensions = {
		CastNarrowingSafe<int32>(SourceTensor.GetSize(1)),
		CastNarrowingSafe<int32>(SourceTensor.GetSize(2)),
		CastNarrowingSafe<int32>(SourceTensor.GetSize(3)),
	};

	// Reusing the same output description for our back buffer as SceneColor
	FRDGTextureDesc DestinationDesc = BaseDestinationDesc;
	// this is flipped because the Output tensor has the vertical dimension first
	// while unreal has the horizontal dimension first
	DestinationDesc.Extent = {SourceTensorDimensions[1], SourceTensorDimensions[0]};
	DestinationDesc.Flags |= TexCreate_RenderTargetable | TexCreate_UAV;
	FLinearColor ClearColor(0., 0., 0., 0.);
	DestinationDesc.ClearValue = FClearValueBinding(ClearColor);
	FRDGTexture* OutputTexture = GraphBuilder.CreateTexture(
		DestinationDesc, TEXT("OutputTexture"));

	FRDGBufferRef SourceTensorBuffer = GraphBuilder.RegisterExternalBuffer(SourceTensor.GetPooledBuffer());

	auto OutputTensorToSceneColorParameters = GraphBuilder.AllocParameters<FOutputTensorToSceneColorCS::FParameters>();
	OutputTensorToSceneColorParameters->InputTensor = GraphBuilder.CreateSRV(SourceTensorBuffer, EPixelFormat::PF_R32_FLOAT);
	OutputTensorToSceneColorParameters->OutputTexture = GraphBuilder.CreateUAV(OutputTexture);
	OutputTensorToSceneColorParameters->TensorVolume = SourceTensor.Num();
	OutputTensorToSceneColorParameters->TextureSize = DestinationDesc.Extent;
	FIntVector OutputTensorToSceneColorGroupCount = FComputeShaderUtils::GetGroupCount(
		{SourceTensorDimensions.X, SourceTensorDimensions.Y, 1},
		FOutputTensorToSceneColorCS::ThreadGroupSize
	);

	TShaderMapRef<FOutputTensorToSceneColorCS> OutputTensorToSceneColorCS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
	GraphBuilder.AddPass(
		RDG_EVENT_NAME("TensorToTexture"),
		OutputTensorToSceneColorParameters,
		ERDGPassFlags::Compute,
		[OutputTensorToSceneColorCS, OutputTensorToSceneColorParameters, OutputTensorToSceneColorGroupCount](FRHICommandList& RHICommandList)
		{
			FComputeShaderUtils::Dispatch(RHICommandList, OutputTensorToSceneColorCS,
			                              *OutputTensorToSceneColorParameters, OutputTensorToSceneColorGroupCount);
		}
	);

	return OutputTexture;
}

void FStyleTransferSceneViewExtension::TextureToTensor(FRDGBuilder& GraphBuilder, FRDGTextureRef SourceTexture, const FNeuralTensor& DestinationTensor)
{
	const FIntVector InputTensorDimensions = {
		CastNarrowingSafe<int32>(DestinationTensor.GetSize(1)),
		CastNarrowingSafe<int32>(DestinationTensor.GetSize(2)),
		CastNarrowingSafe<int32>(DestinationTensor.GetSize(3)),
	};
	const FIntPoint SceneColorRenderTargetDimensions = SourceTexture->Desc.Extent;

	FRDGBufferRef StyleTransferContentInputBuffer = GraphBuilder.RegisterExternalBuffer(DestinationTensor.GetPooledBuffer());
	auto SceneColorToInputTensorParameters = GraphBuilder.AllocParameters<FSceneColorToInputTensorCS::FParameters>();
	SceneColorToInputTensorParameters->TensorVolume = CastNarrowingSafe<uint32>(DestinationTensor.Num());
	SceneColorToInputTensorParameters->InputTexture = SourceTexture;
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
		RDG_EVENT_NAME("TextureToTensor"),
		SceneColorToInputTensorParameters,
		ERDGPassFlags::Compute,
		[SceneColorToInputTensorCS, SceneColorToInputTensorParameters, SceneColorToInputTensorGroupCount](FRHICommandList& RHICommandList)
		{
			FComputeShaderUtils::Dispatch(RHICommandList, SceneColorToInputTensorCS,
			                              *SceneColorToInputTensorParameters, SceneColorToInputTensorGroupCount);
		}
	);
}

FScreenPassTexture FStyleTransferSceneViewExtension::PostProcessPassAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& InOutInputs)
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

	checkSlow(View.bIsViewInfo);
	const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);

	const FNeuralTensor& StyleTransferContentInputTensor = StyleTransferNetwork->GetInputTensorForContext(*InferenceContext, 0);

	TextureToTensor(GraphBuilder, SceneColor.Texture, StyleTransferContentInputTensor);

	StyleTransferNetwork->Run(GraphBuilder, *InferenceContext);

	const FNeuralTensor& StyleTransferContentOutputTensor = StyleTransferNetwork->GetOutputTensorForContext(*InferenceContext, 0);
	FRDGTexture* StyleTransferRenderTargetTexture = TensorToTexture(GraphBuilder, SceneColor.Texture->Desc, StyleTransferContentOutputTensor);

	TSharedPtr<FScreenPassRenderTarget> StyleTransferOutputTarget = MakeShared<FScreenPassRenderTarget>(StyleTransferRenderTargetTexture, SceneColor.ViewRect,
	                                                                                                    ERenderTargetLoadAction::EClear);


	TSharedPtr<FScreenPassRenderTarget> BackBufferRenderTarget;
	// If the override output is provided it means that this is the last pass in post processing.
	if (InOutInputs.OverrideOutput.IsValid())
	{
		BackBufferRenderTarget = MakeShared<FScreenPassRenderTarget>(InOutInputs.OverrideOutput);

		AddRescalingTextureCopy(GraphBuilder, *StyleTransferOutputTarget->Texture, *BackBufferRenderTarget);
	}
	else
	{
		BackBufferRenderTarget = StyleTransferOutputTarget;
	}

	return MoveTemp(*BackBufferRenderTarget);
}
