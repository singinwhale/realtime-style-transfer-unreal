// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "IRenderCaptureProvider.h"
#include "StyleTransferSceneViewExtension.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "UObject/Object.h"
#include "StyleTransferSubsystem.generated.h"

/**
 *
 */
UCLASS()
class STYLETRANSFER_API UStyleTransferSubsystem : public UGameInstanceSubsystem, public FTSTickerObjectBase
{
	GENERATED_BODY()

public:
	// - UGameInstanceSubsystem
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	// --

	// - FTSTickerObjectBase
	virtual bool Tick(float DeltaTime) override final;
	// --

	void StartStylizingViewport(FViewportClient* ViewportClient);
	void StopStylizingViewport();

	void UpdateStyle(UTexture2D* StyleTexture, uint32 StyleIndex, int32 StylePredictionInferenceContext);
	void UpdateStyle(FString StyleTensorDataPath);
	void InterpolateStyles(int32 StylePredictionInferenceContextA, int32 StylePredictionInferenceContextB, float Alpha);

private:
	FStyleTransferSceneViewExtension::Ptr StyleTransferSceneViewExtension;

	UPROPERTY()
	TObjectPtr<UNeuralNetwork> StyleTransferNetwork;

	UPROPERTY()
	TObjectPtr<UNeuralNetwork> StylePredictionNetwork;

	TArray<int32> StylePredictionInferenceContexts;
	TSharedPtr<int32, ESPMode::ThreadSafe> StyleTransferInferenceContext;


	int32 StyleTransferStyleParamsInputIndex = INDEX_NONE;

	void HandleConsoleVariableChanged(IConsoleVariable*);

	void LoadNetworks();
};

IRenderCaptureProvider* BeginRenderCapture(FRHICommandListImmediate& RHICommandList);
IRenderCaptureProvider* ConditionalBeginRenderCapture(FRHICommandListImmediate& RHICommandList);