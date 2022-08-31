// Fill out your copyright notice in the Description page of Project Settings.


#include "StyleTransferSubsystem.h"

#include "NeuralNetwork.h"
#include "RenderGraphUtils.h"
#include "ScreenPass.h"
#include "StyleTransferModule.h"
#include "StyleTransferSceneViewExtension.h"
#include "Rendering/Texture2DResource.h"

TAutoConsoleVariable<bool> CVarStyleTransferEnabled(
	TEXT("r.StyleTransfer.Enabled"),
	false,
	TEXT("Set to true to enable style transfer")
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

void UStyleTransferSubsystem::StartStylizingViewport(FViewportClient* ViewportClient)
{
	if (!StylePredictionNetwork->IsLoaded() || !StyleTransferNetwork->IsLoaded())
	{
		UE_LOG(LogStyleTransfer, Error, TEXT("Not all networks were loaded, can not stylize viewport."));
		return;
	}

	if (!StyleTransferSceneViewExtension)
	{
		StylePredictionInferenceContext = StylePredictionNetwork->CreateInferenceContext();
		checkf(StylePredictionInferenceContext != INDEX_NONE, TEXT("Could not create inference context for StylePredictionNetwork"));
		StyleTransferInferenceContext = MakeShared<int32>(StyleTransferNetwork->CreateInferenceContext());
		checkf(*StyleTransferInferenceContext != INDEX_NONE, TEXT("Could not create inference context for StyleTransferNetwork"));

		UTexture2D* StyleTexture = LoadObject<UTexture2D>(this, TEXT("/Script/Engine.Texture2D'/StyleTransfer/T_StyleImage.T_StyleImage'"));
		UpdateStyle(StyleTexture);
		StyleTransferSceneViewExtension = FSceneViewExtensions::NewExtension<FStyleTransferSceneViewExtension>(ViewportClient, StyleTransferNetwork, StyleTransferInferenceContext.ToSharedRef());

	}
	StyleTransferSceneViewExtension->SetEnabled(true);
}

void UStyleTransferSubsystem::StopStylizingViewport()
{
	FlushRenderingCommands();
	StyleTransferSceneViewExtension.Reset();
	if (StylePredictionInferenceContext != INDEX_NONE)
	{
		StylePredictionNetwork->DestroyInferenceContext(StylePredictionInferenceContext);
	}
	if (StyleTransferInferenceContext && *StyleTransferInferenceContext != INDEX_NONE)
	{
		StyleTransferNetwork->DestroyInferenceContext(*StyleTransferInferenceContext);
		*StyleTransferInferenceContext = INDEX_NONE;
		StyleTransferInferenceContext.Reset();
	}
}

void UStyleTransferSubsystem::UpdateStyle(UTexture2D* StyleTexture)
{
	checkf(StyleTransferInferenceContext.IsValid() && (*StyleTransferInferenceContext) != INDEX_NONE, TEXT("Can not infer style without inference context"));
	checkf(StylePredictionInferenceContext != INDEX_NONE, TEXT("Can not update style without inference context"));

	ENQUEUE_RENDER_COMMAND(StylePrediction)([this, StyleTexture](FRHICommandListImmediate& RHICommandList)
	{
		FRDGBuilder GraphBuilder(RHICommandList);


		const FNeuralTensor& InputStyleImageTensor = StylePredictionNetwork->GetInputTensorForContext(StylePredictionInferenceContext, 0);

		FTextureResource* StyleTextureResource = StyleTexture->GetResource();
		if(!StyleTextureResource->IsInitialized())
		{
			StyleTextureResource->UpdateRHI();
		}

		FRDGTextureRef RDGStyleTexture = GraphBuilder.RegisterExternalTexture(CreateRenderTarget(StyleTextureResource->TextureRHI, TEXT("StyleInputTexture")));
		FStyleTransferSceneViewExtension::TextureToTensor(GraphBuilder, FScreenPassTexture(RDGStyleTexture), InputStyleImageTensor);

		StylePredictionNetwork->Run(GraphBuilder, StylePredictionInferenceContext);

		const FNeuralTensor& OutputStyleParams = StylePredictionNetwork->GetOutputTensorForContext(StylePredictionInferenceContext, 0);
		const FNeuralTensor& InputStyleParams = StyleTransferNetwork->GetInputTensorForContext(*StyleTransferInferenceContext, StyleTransferStyleParamsInputIndex);

		FRDGBufferRef OutputStyleParamsBuffer = GraphBuilder.RegisterExternalBuffer(OutputStyleParams.GetPooledBuffer());
		FRDGBufferRef InputStyleParamsBuffer = GraphBuilder.RegisterExternalBuffer(InputStyleParams.GetPooledBuffer());
		const uint64 NumBytes = OutputStyleParams.NumInBytes();
		check(OutputStyleParamsBuffer->GetSize() == InputStyleParamsBuffer->GetSize());
		check(OutputStyleParamsBuffer->GetSize() == OutputStyleParams.NumInBytes());
		check(InputStyleParamsBuffer->GetSize() == InputStyleParams.NumInBytes());

		AddCopyBufferPass(GraphBuilder, InputStyleParamsBuffer, OutputStyleParamsBuffer);

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
		if(!(StyleTransferNetwork || StylePredictionNetwork))
		{
			LoadNetworks();
		}
		StartStylizingViewport(GetGameInstance()->GetGameViewportClient());
	}
}

void UStyleTransferSubsystem::LoadNetworks()
{
	StyleTransferNetwork = LoadObject<UNeuralNetwork>(this, TEXT("/StyleTransfer/NN_StyleTransfer.NN_StyleTransfer"));
	StylePredictionNetwork = LoadObject<UNeuralNetwork>(this, TEXT("/StyleTransfer/NN_StylePredictor.NN_StylePredictor"));

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
