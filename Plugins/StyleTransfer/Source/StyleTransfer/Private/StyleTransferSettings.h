// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "NeuralNetwork.h"
#include "UObject/Object.h"
#include "StyleTransferSettings.generated.h"

/**
 *
 */
UCLASS(Config=Game, defaultconfig, meta=(DisplayName="Style Transfer"))
class STYLETRANSFER_API UStyleTransferSettings : public UDeveloperSettings
{
	GENERATED_BODY()
public:
	UStyleTransferSettings();

	UPROPERTY(EditAnywhere, Config)
	TSoftObjectPtr<UNeuralNetwork> StyleTransferNetwork = nullptr;

	UPROPERTY(EditAnywhere, Config)
	TSoftObjectPtr<UNeuralNetwork> StylePredictionNetwork = nullptr;

	UPROPERTY(EditAnywhere, Config)
	TSoftObjectPtr<UTexture2D> StyleTexture = nullptr;
};
