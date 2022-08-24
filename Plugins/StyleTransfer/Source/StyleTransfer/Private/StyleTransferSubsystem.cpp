// Fill out your copyright notice in the Description page of Project Settings.


#include "StyleTransferSubsystem.h"

#include "NeuralNetwork.h"
#include "RenderGraphUtils.h"
#include "StyleTransferSceneViewExtension.h"

void UStyleTransferSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

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
		UE_LOG(LogTemp, Warning, TEXT("StyleTransferNetwork could not be loaded"));
	}


	if (StylePredictionNetwork->IsLoaded())
	{
		StyleTransferNetwork->SetDeviceType(ENeuralDeviceType::GPU, ENeuralDeviceType::GPU, ENeuralDeviceType::GPU);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StylePredictionNetwork could not be loaded."));
	}
}

void UStyleTransferSubsystem::Deinitialize()
{
	StyleTransferSceneViewExtension.Reset();
	if(StylePredictionInferenceContext != INDEX_NONE)
	{
		StylePredictionNetwork->DestroyInferenceContext(StylePredictionInferenceContext);
	}
	if(StyleTransferInferenceContext != INDEX_NONE)
	{
		StyleTransferNetwork->DestroyInferenceContext(StyleTransferInferenceContext);
	}

	Super::Deinitialize();
}

void UStyleTransferSubsystem::StartStylizingViewport(FViewportClient* ViewportClient)
{
	StylePredictionInferenceContext = StylePredictionNetwork->CreateInferenceContext();
	checkf(StylePredictionInferenceContext != INDEX_NONE, TEXT("Could not create inference context for StylePredictionNetwork"));
	StyleTransferInferenceContext = StyleTransferNetwork->CreateInferenceContext();
	checkf(StyleTransferInferenceContext != INDEX_NONE, TEXT("Could not create inference context for StyleTransferNetwork"));

	StyleTransferSceneViewExtension = FSceneViewExtensions::NewExtension<FStyleTransferSceneViewExtension>(ViewportClient, StyleTransferNetwork, StyleTransferInferenceContext);
}

void UStyleTransferSubsystem::UpdateStyle(const FNeuralTensor& StyleImage)
{
	checkf(StyleTransferSceneViewExtension.IsValid(), TEXT("Can not update style while not stylizing"));

	StylePredictionNetwork->SetInputFromArrayCopy(StyleImage.GetArrayCopy<float>());

	ENQUEUE_RENDER_COMMAND(StylePrediction)([this](FRHICommandListImmediate& RHICommandList)
	{
		FRDGBuilder GraphBuilder(RHICommandList);

		StylePredictionNetwork->Run(GraphBuilder, StylePredictionInferenceContext);

		const FNeuralTensor& OutputStyleParams = StylePredictionNetwork->GetOutputTensorForContext(StylePredictionInferenceContext, 0);
		const FNeuralTensor& InputStyleParams = StyleTransferNetwork->GetInputTensorForContext(StyleTransferInferenceContext, StyleTransferStyleParamsInputIndex);

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
