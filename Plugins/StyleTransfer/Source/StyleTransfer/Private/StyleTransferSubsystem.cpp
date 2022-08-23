// Fill out your copyright notice in the Description page of Project Settings.


#include "StyleTransferSubsystem.h"

#include "NeuralNetwork.h"
#include "StyleTransferSceneViewExtension.h"

void UStyleTransferSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	StyleTransferNetwork = NewObject<UNeuralNetwork>();

	FString ONNXModelFilePath = TEXT("SOME_PARENT_FOLDER/SOME_ONNX_FILE_NAME.onnx");
	if (StyleTransferNetwork->Load(ONNXModelFilePath))
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
		UE_LOG(LogTemp, Warning, TEXT("StyleTransferNetwork could not loaded from %s."), *ONNXModelFilePath);
	}

	StylePredictionNetwork = NewObject<UNeuralNetwork>();

	ONNXModelFilePath = TEXT("SOME_PARENT_FOLDER/SOME_ONNX_FILE_NAME.onnx");
	if (StylePredictionNetwork->Load(ONNXModelFilePath))
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
		UE_LOG(LogTemp, Warning, TEXT("StylePredictionNetwork could not loaded from %s."), *ONNXModelFilePath);
	}
}

void UStyleTransferSubsystem::StartStylizingViewport(FViewportClient* ViewportClient)
{
	StyleTransferSceneViewExtension = FSceneViewExtensions::NewExtension<FStyleTransferSceneViewExtension>(ViewportClient, StyleTransferNetwork);
}

void UStyleTransferSubsystem::UpdateStyle(FNeuralTensor StyleImage)
{

}
