// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "StyleTransferSceneViewExtension.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "UObject/Object.h"
#include "StyleTransferSubsystem.generated.h"

/**
 *
 */
UCLASS()
class STYLETRANSFER_API UStyleTransferSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	// - UGameInstanceSubsystem
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	// --

	void StartStylizingViewport(FViewportClient* ViewportClient);
	void StopStylizingViewport();

	void UpdateStyle(UTexture2D* StyleTexture);
	void UpdateStyle(FString StyleTensorDataPath);

private:
	FStyleTransferSceneViewExtension::Ptr StyleTransferSceneViewExtension;

	UPROPERTY()
	TObjectPtr<UNeuralNetwork> StyleTransferNetwork;

	UPROPERTY()
	TObjectPtr<UNeuralNetwork> StylePredictionNetwork;

	int32 StylePredictionInferenceContext = INDEX_NONE;
	TSharedPtr<int32, ESPMode::ThreadSafe> StyleTransferInferenceContext;


	int32 StyleTransferStyleParamsInputIndex = INDEX_NONE;

	void HandleConsoleVariableChanged(IConsoleVariable*);

	void LoadNetworks();
};
