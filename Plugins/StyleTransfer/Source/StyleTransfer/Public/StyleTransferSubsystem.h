// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "NeuralTensor.h"
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

	void UpdateStyle(const FNeuralTensor& StyleImage);

private:
	FStyleTransferSceneViewExtension::Ptr StyleTransferSceneViewExtension;

	UPROPERTY()
	TObjectPtr<UNeuralNetwork> StyleTransferNetwork;

	UPROPERTY()
	TObjectPtr<UNeuralNetwork> StylePredictionNetwork;

	int32 StylePredictionInferenceContext = INDEX_NONE;
	int32 StyleTransferInferenceContext = INDEX_NONE;

	int32 StyleTransferStyleParamsInputIndex = INDEX_NONE;
};