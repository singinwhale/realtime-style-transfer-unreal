// Fill out your copyright notice in the Description page of Project Settings.


#include "StyleTransferSubsystem.h"

#include "IRenderCaptureProvider.h"
#include "NeuralNetwork.h"
#include "RenderGraphUtils.h"
#include "ScreenPass.h"
#include "StyleTransferModule.h"
#include "StyleTransferSceneViewExtension.h"
#include "StyleTransferSettings.h"
#include "TextureCompiler.h"
#include "Rendering/Texture2DResource.h"

TAutoConsoleVariable<bool> CVarStyleTransferEnabled(
	TEXT("r.StyleTransfer.Enabled"),
	false,
	TEXT("Set to true to enable style transfer")
);

TAutoConsoleVariable<bool> CVarAutoCaptureStylePrediction(
	TEXT("r.StyleTransfer.AutoCapturePrediction"),
	false,
	TEXT("Set to true to enable style transfer auto capture for profiling in PIX etc.")
);


void UStyleTransferSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	CVarStyleTransferEnabled->OnChangedDelegate().AddUObject(this, &UStyleTransferSubsystem::HandleConsoleVariableChanged);
}

void UStyleTransferSubsystem::Deinitialize()
{
	StopStylizingViewport();

	Super::Deinitialize();
}

bool UStyleTransferSubsystem::Tick(float DeltaTime)
{
	return false;
	if (!GetWorld())
		return true;


	if (StylePredictionInferenceContexts.Num() > 1)
	{
		const UStyleTransferSettings* StyleTransferSettings = GetDefault<UStyleTransferSettings>();
		const FRichCurve* InterpCurve = StyleTransferSettings->InterpolationCurve.GetRichCurveConst();
		float MinTime, MaxTime;
		InterpCurve->GetTimeRange(MinTime, MaxTime);
		const double Time = MinTime + FMath::Fmod(GetWorld()->GetTimeSeconds(), static_cast<double>(MaxTime - MinTime));
		const float Alpha = InterpCurve->Eval(Time);
		UE_LOG(LogStyleTransfer, VeryVerbose, TEXT("Alpha is %0.4f"), Alpha);
		InterpolateStyles(StylePredictionInferenceContexts[0], StylePredictionInferenceContexts[1], Alpha);
	}
	return true;
}

void UStyleTransferSubsystem::StartStylizingViewport(FViewportClient* ViewportClient)
{
	if (!StylePredictionNetwork->IsLoaded() || !StyleTransferNetwork->IsLoaded())
	{
		UE_LOG(LogStyleTransfer, Error, TEXT("Not all networks were loaded, can not stylize viewport."));
		return;
	}

	if (!StyleTransferSceneViewExtension)
	{
		const UStyleTransferSettings* StyleTransferSettings = GetDefault<UStyleTransferSettings>();

		if (!StyleTransferInferenceContext || *StyleTransferInferenceContext == INDEX_NONE)
		{
			UE_LOG(LogStyleTransfer, Log, TEXT("Creating Inference Context for StyleTransfer"));
			StyleTransferInferenceContext = MakeShared<int32>(StyleTransferNetwork->CreateInferenceContext());
			checkf(*StyleTransferInferenceContext != INDEX_NONE, TEXT("Could not create inference context for StyleTransferNetwork"));
		}

		for (uint32 i = 0; i < FMath::Min(2u, uint32(StyleTransferSettings->StyleTextures.Num())); ++i)
		{
			UE_LOG(LogStyleTransfer, Log, TEXT("Creating Inference Context for Style %i"), i);
			const int32& StylePredictionInferenceContext = StylePredictionInferenceContexts.Emplace_GetRef(StylePredictionNetwork->CreateInferenceContext());
			checkf(StylePredictionInferenceContext != INDEX_NONE, TEXT("Could not create inference context for StylePredictionNetwork"));

			UTexture2D* StyleTexture = StyleTransferSettings->StyleTextures[i].LoadSynchronous();
			//UTexture2D* StyleTexture = LoadObject<UTexture2D>(this, TEXT("/Script/Engine.Texture2D'/StyleTransfer/T_StyleImage.T_StyleImage'"));
#if WITH_EDITOR
			FTextureCompilingManager::Get().FinishCompilation({StyleTexture});
#endif
			UpdateStyle(StyleTexture, i, StylePredictionInferenceContext);
		}
		//UpdateStyle(FPaths::GetPath("C:\\projects\\realtime-style-transfer\\temp\\style_params_tensor.bin"));
		UE_LOG(LogStyleTransfer, Log, TEXT("Creating FStyleTransferSceneViewExtension"));
		StyleTransferSceneViewExtension = FSceneViewExtensions::NewExtension<FStyleTransferSceneViewExtension>(ViewportClient->GetWorld(), ViewportClient, StyleTransferNetwork, StyleTransferInferenceContext.ToSharedRef());
	}
	if (StyleTransferSceneViewExtension)
	{
		StyleTransferSceneViewExtension->SetEnabled(true);
	}
}

void UStyleTransferSubsystem::StopStylizingViewport()
{
	FlushRenderingCommands();
	StyleTransferSceneViewExtension.Reset();
	if (StylePredictionInferenceContexts.Num())
	{
		for (auto It = StylePredictionInferenceContexts.CreateIterator(); It; ++It)
		{
			StylePredictionNetwork->DestroyInferenceContext(*It);
			It.RemoveCurrent();
		}
	}
	if (StyleTransferInferenceContext && *StyleTransferInferenceContext != INDEX_NONE)
	{
		StyleTransferNetwork->DestroyInferenceContext(*StyleTransferInferenceContext);
		*StyleTransferInferenceContext = INDEX_NONE;
		StyleTransferInferenceContext.Reset();
	}
}

BEGIN_SHADER_PARAMETER_STRUCT(FCopyBufferParameters,)
	RDG_BUFFER_ACCESS(SrcBuffer, ERHIAccess::CopySrc)
	RDG_BUFFER_ACCESS(DstBuffer, ERHIAccess::CopyDest)
END_SHADER_PARAMETER_STRUCT()

void UStyleTransferSubsystem::UpdateStyle(UTexture2D* StyleTexture, uint32 StyleIndex, int32 StylePredictionInferenceContext)
{
	checkf(StyleTransferInferenceContext.IsValid() && (*StyleTransferInferenceContext) != INDEX_NONE, TEXT("Can not infer style without inference context"));
	checkf(StylePredictionInferenceContext != INDEX_NONE, TEXT("Can not update style without inference context"));
	FlushRenderingCommands();
	ENQUEUE_RENDER_COMMAND(StylePrediction)([this, StyleTexture, StylePredictionInferenceContext, StyleIndex](FRHICommandListImmediate& RHICommandList)
	{
		IRenderCaptureProvider* RenderCaptureProvider = ConditionalBeginRenderCapture(RHICommandList);

		FRDGBuilder GraphBuilder(RHICommandList);
		{
			RDG_EVENT_SCOPE(GraphBuilder, "StylePrediction");

			const FNeuralTensor& InputStyleImageTensor = StylePredictionNetwork->GetInputTensorForContext(StylePredictionInferenceContext, 0);
			FTextureResource* StyleTextureResource = StyleTexture->GetResource();
			FRDGTextureRef RDGStyleTexture = GraphBuilder.RegisterExternalTexture(CreateRenderTarget(StyleTextureResource->TextureRHI, TEXT("StyleInputTexture")));
			FStyleTransferSceneViewExtension::TextureToTensor(GraphBuilder, RDGStyleTexture, InputStyleImageTensor);

			StylePredictionNetwork->Run(GraphBuilder, StylePredictionInferenceContext);

			const FNeuralTensor& OutputStyleParams = StylePredictionNetwork->GetOutputTensorForContext(StylePredictionInferenceContext, 0);
			const FNeuralTensor& InputStyleParams = StyleTransferNetwork->GetInputTensorForContext(*StyleTransferInferenceContext, StyleTransferStyleParamsInputIndex);

			FRDGBufferRef OutputStyleParamsBuffer = GraphBuilder.RegisterExternalBuffer(OutputStyleParams.GetPooledBuffer());
			FRDGBufferRef InputStyleParamsBuffer = GraphBuilder.RegisterExternalBuffer(InputStyleParams.GetPooledBuffer());
			const uint64 NumBytes = OutputStyleParams.NumInBytes();
			const uint64 DstOffset = StyleIndex * NumBytes;


			FCopyBufferParameters* Parameters = GraphBuilder.AllocParameters<FCopyBufferParameters>();
			Parameters->SrcBuffer = OutputStyleParamsBuffer;
			Parameters->DstBuffer = InputStyleParamsBuffer;

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("CopyBuffer(%s Size=%ubytes)", Parameters->SrcBuffer, Parameters->SrcBuffer->Desc.GetSize()),
				Parameters,
				ERDGPassFlags::Copy,
				[Parameters, NumBytes, DstOffset](FRHICommandList& RHICmdList)
				{
					RHICmdList.CopyBufferRegion(Parameters->DstBuffer->GetRHI(), DstOffset, Parameters->SrcBuffer->GetRHI(), 0, NumBytes);
				});
		}
		GraphBuilder.Execute();

		if (RenderCaptureProvider)
		{
			RenderCaptureProvider->EndCapture(&RHICommandList);
		}
	});

	FlushRenderingCommands();
}

void UStyleTransferSubsystem::UpdateStyle(FString StyleTensorDataPath)
{
	FArchive& FileReader = *IFileManager::Get().CreateFileReader(*StyleTensorDataPath);
	TArray<float> StyleParams;
	StyleParams.SetNumUninitialized(2758);

	FileReader << StyleParams;

	ENQUEUE_RENDER_COMMAND(StyleParamsLoad)([this, StyleParams = MoveTemp(StyleParams)](FRHICommandListImmediate& RHICommandList)
	{
		const FNeuralTensor& InputStyleParams = StyleTransferNetwork->GetInputTensorForContext(*StyleTransferInferenceContext, StyleTransferStyleParamsInputIndex);

		FRDGBuilder GraphBuilder(RHICommandList);
		{
			RDG_EVENT_SCOPE(GraphBuilder, "StyleParamsLoad");

			FRDGBufferRef InputStyleParamsBuffer = GraphBuilder.RegisterExternalBuffer(InputStyleParams.GetPooledBuffer());
			GraphBuilder.QueueBufferUpload(InputStyleParamsBuffer, StyleParams.GetData(), StyleParams.Num() * StyleParams.GetTypeSize(), ERDGInitialDataFlags::NoCopy);
		}
		GraphBuilder.Execute();
	});
	FlushRenderingCommands();
}

void UStyleTransferSubsystem::HandleConsoleVariableChanged(IConsoleVariable* ConsoleVariable)
{
	check(ConsoleVariable == CVarStyleTransferEnabled.AsVariable());

	StopStylizingViewport();

	if (CVarStyleTransferEnabled->GetBool())
	{
		if (!(StyleTransferNetwork || StylePredictionNetwork))
		{
			LoadNetworks();
		}
		StartStylizingViewport(GetGameInstance()->GetGameViewportClient());
	}
}

void UStyleTransferSubsystem::LoadNetworks()
{
	const UStyleTransferSettings* StyleTransferSettings = GetDefault<UStyleTransferSettings>();
	StyleTransferNetwork = StyleTransferSettings->StyleTransferNetwork.LoadSynchronous();
	StylePredictionNetwork = StyleTransferSettings->StylePredictionNetwork.LoadSynchronous();

	if (StyleTransferNetwork->IsLoaded())
	{
		for (int32 i = 0; i < StyleTransferNetwork->GetInputTensorNumber(); ++i)
		{
			const FNeuralTensor& InputTensor = StyleTransferNetwork->GetInputTensor(i);
			if (InputTensor.GetName() != "style_params")
				continue;

			StyleTransferStyleParamsInputIndex = i;
			break;
		}
		StyleTransferNetwork->SetDeviceType(ENeuralDeviceType::GPU, ENeuralDeviceType::GPU, ENeuralDeviceType::GPU);
	}
	else
	{
		UE_LOG(LogStyleTransfer, Error, TEXT("StyleTransferNetwork could not be loaded"));
	}


	if (StylePredictionNetwork->IsLoaded())
	{
		StyleTransferNetwork->SetDeviceType(ENeuralDeviceType::GPU, ENeuralDeviceType::GPU, ENeuralDeviceType::GPU);
	}
	else
	{
		UE_LOG(LogStyleTransfer, Error, TEXT("StylePredictionNetwork could not be loaded."));
	}
}

void UStyleTransferSubsystem::InterpolateStyles(int32 StylePredictionInferenceContextA, int32 StylePredictionInferenceContextB, float Alpha)
{
	checkf(StyleTransferInferenceContext.IsValid() && (*StyleTransferInferenceContext) != INDEX_NONE, TEXT("Can not transfer style without inference context"));
	checkf(StylePredictionInferenceContexts.Contains(StylePredictionInferenceContextA), TEXT("Can not update style without inference context A"));
	checkf(StylePredictionInferenceContexts.Contains(StylePredictionInferenceContextB), TEXT("Can not update style without inference context B"));
	ENQUEUE_RENDER_COMMAND(StylePrediction)([this, StylePredictionInferenceContextA, StylePredictionInferenceContextB, Alpha](FRHICommandListImmediate& RHICommandList)
	{
		IRenderCaptureProvider* RenderCaptureProvider = ConditionalBeginRenderCapture(RHICommandList);
		FRDGBuilder GraphBuilder(RHICommandList);
		{
			RDG_EVENT_SCOPE(GraphBuilder, "StylePrediction");

			const FNeuralTensor& InputStyleImageTensorA = StylePredictionNetwork->GetOutputTensorForContext(StylePredictionInferenceContextA, 0);
			const FNeuralTensor& InputStyleImageTensorB = StylePredictionNetwork->GetOutputTensorForContext(StylePredictionInferenceContextB, 0);
			const FNeuralTensor& OutputStyleParamsTensor = StyleTransferNetwork->GetInputTensorForContext(*StyleTransferInferenceContext, StyleTransferStyleParamsInputIndex);
			FStyleTransferSceneViewExtension::InterpolateTensors(GraphBuilder, OutputStyleParamsTensor, InputStyleImageTensorA, InputStyleImageTensorB, Alpha);
		}
		GraphBuilder.Execute();
		if (RenderCaptureProvider) RenderCaptureProvider->EndCapture(&RHICommandList);
	});
}

IRenderCaptureProvider* BeginRenderCapture(FRHICommandListImmediate& RHICommandList)
{
	IRenderCaptureProvider* RenderCaptureProvider = nullptr;
	const FName RenderCaptureProviderType = IRenderCaptureProvider::GetModularFeatureName();
	if (IModularFeatures::Get().IsModularFeatureAvailable(RenderCaptureProviderType))
	{
		RenderCaptureProvider = &IModularFeatures::Get().GetModularFeature<IRenderCaptureProvider>(RenderCaptureProviderType);
		RenderCaptureProvider->EndCapture(&RHICommandList);
		RenderCaptureProvider->BeginCapture(&RHICommandList);
	}
	return RenderCaptureProvider;
}

IRenderCaptureProvider* ConditionalBeginRenderCapture(FRHICommandListImmediate& RHICommandList)
{
	IRenderCaptureProvider* RenderCaptureProvider = nullptr;
	if (CVarAutoCaptureStylePrediction.GetValueOnRenderThread())
	{
		RenderCaptureProvider = BeginRenderCapture(RHICommandList);
	}
	return RenderCaptureProvider;
}
