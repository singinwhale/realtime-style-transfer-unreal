// Fill out your copyright notice in the Description page of Project Settings.


#include "StyleTransferSubsystem.h"

#include "NeuralNetwork.h"
#include "StyleTransferSceneViewExtension.h"

void UStyleTransferSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	StyleTransferNetwork = LoadObject<UNeuralNetwork>(this, TEXT("/StyleTransfer/NN_StyleTransfer.NN_StyleTransfer"));
	StylePredictionNetwork = LoadObject<UNeuralNetwork>(this, TEXT("/StyleTransfer/NN_StylePredictor.NN_StylePredictor"));

	if (StyleTransferNetwork->IsLoaded())
	{
		if (StyleTransferNetwork->IsGPUSupported())
		{
			StyleTransferNetwork->SetDeviceType(ENeuralDeviceType::GPU);
		}
		else
		{
			StyleTransferNetwork->SetDeviceType(ENeuralDeviceType::CPU);
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StyleTransferNetwork could not be loaded"));
	}


	if (StylePredictionNetwork->IsLoaded())
	{
		if (StylePredictionNetwork->IsGPUSupported())
		{
			StylePredictionNetwork->SetDeviceType(ENeuralDeviceType::GPU);
		}
		else
		{
			StylePredictionNetwork->SetDeviceType(ENeuralDeviceType::CPU);
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StylePredictionNetwork could not be loaded."));
	}
}

void UStyleTransferSubsystem::Deinitialize()
{
	StyleTransferSceneViewExtension.Reset();
	StylePredictionNetwork->DestroyInferenceContext(StylePredictionInferenceContext);

	Super::Deinitialize();
}

void UStyleTransferSubsystem::StartStylizingViewport(FViewportClient* ViewportClient)
{
	StyleTransferSceneViewExtension = FSceneViewExtensions::NewExtension<FStyleTransferSceneViewExtension>(ViewportClient, StyleTransferNetwork);
	StylePredictionInferenceContext = StylePredictionNetwork->CreateInferenceContext();
}

void UStyleTransferSubsystem::UpdateStyle(FNeuralTensor StyleImage)
{
	StylePredictionNetwork->SetInputFromArrayCopy(StyleImage.GetArrayCopy<float>());

	ENQUEUE_RENDER_COMMAND(StylePrediction)([this](FRHICommandListImmediate& RHICommandList)
	{
		FRDGBuilder GraphBuilder(RHICommandList);

		StylePredictionNetwork->Run(GraphBuilder, StylePredictionInferenceContext);

		// @todo: copy output of style prediction network to input of style transfer network

		GraphBuilder.Execute();
	});

	FlushRenderingCommands();
}
