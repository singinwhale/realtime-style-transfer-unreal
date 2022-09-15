#pragma once
#include "SceneViewExtension.h"

struct FNeuralTensor;
struct FScreenPassRenderTarget;
class UNeuralNetwork;

class FStyleTransferSceneViewExtension : public FWorldSceneViewExtension
{
public:
	using Ptr = TSharedPtr<FStyleTransferSceneViewExtension, ESPMode::ThreadSafe>;
	using Ref = TSharedRef<FStyleTransferSceneViewExtension, ESPMode::ThreadSafe>;

	FStyleTransferSceneViewExtension(const FAutoRegister& AutoRegister, UWorld* World, FViewportClient* AssociatedViewportClient, UNeuralNetwork* InStyleTransferNetwork, TSharedRef<int32> InInferenceContext);

	// - ISceneViewExtension
	virtual void SubscribeToPostProcessingPass(EPostProcessingPass Pass, FAfterPassCallbackDelegateArray& InOutPassCallbacks, bool bIsPassEnabled) override;

	virtual void PreRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily) override;

	FScreenPassTexture PostProcessPassAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View,
	                                                            const FPostProcessMaterialInputs& InOutInputs);

	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override
	{
	}

	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override
	{
	}

	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override
	{
	}

	virtual bool IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const override;
	// --

	void SetEnabled(bool bInIsEnabled) { bIsEnabled = bInIsEnabled; }
	bool IsEnabled() const { return bIsEnabled; }


	static void AddRescalingTextureCopy(FRDGBuilder& GraphBuilder, FRDGTexture& RDGSourceTexture, FScreenPassRenderTarget& DestinationRenderTarget);
	static FRDGTexture* TensorToTexture(FRDGBuilder& GraphBuilder, const FRDGTextureDesc& BaseDestinationDesc, const FNeuralTensor& SourceTensor);
	static void TextureToTensor(FRDGBuilder& GraphBuilder, FRDGTextureRef SourceTexture, const FNeuralTensor& DestinationTensor);
	static void InterpolateTensors(FRDGBuilder& GraphBuilder, const FNeuralTensor& DestinationTensor, const FNeuralTensor& InputTensorA, const FNeuralTensor& InputTensorB, float Alpha);

private:
	/** The actual Network pointer is not tracked so we need a WeakPtr too so we can check its validity on the game thread. */
	TWeakObjectPtr<UNeuralNetwork> StyleTransferNetworkWeakPtr;
	TObjectPtr<UNeuralNetwork> StyleTransferNetwork;

	FViewportClient* LinkedViewportClient;

	TSharedRef<int32, ESPMode::ThreadSafe> InferenceContext = MakeShared<int32>(-1);

	bool bIsEnabled = true;

	int32 NumFramesCaptured = -1;
};
