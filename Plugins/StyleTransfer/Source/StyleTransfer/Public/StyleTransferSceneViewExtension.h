#pragma once
#include "SceneViewExtension.h"

struct FNeuralTensor;
struct FScreenPassRenderTarget;
class UNeuralNetwork;

class FStyleTransferSceneViewExtension : public FSceneViewExtensionBase
{
public:
	using Ptr = TSharedPtr<FStyleTransferSceneViewExtension, ESPMode::ThreadSafe>;
	using Ref = TSharedRef<FStyleTransferSceneViewExtension, ESPMode::ThreadSafe>;

	FStyleTransferSceneViewExtension(const FAutoRegister& AutoRegister, FViewportClient* AssociatedViewportClient, UNeuralNetwork* InStyleTransferNetwork, TSharedRef<int32> InInferenceContext);

	// - ISceneViewExtension
	virtual void SubscribeToPostProcessingPass(EPostProcessingPass Pass, FAfterPassCallbackDelegateArray& InOutPassCallbacks, bool bIsPassEnabled) override;
	FScreenPassTexture PostProcessPassAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View,
	                                                            const FPostProcessMaterialInputs& InOutInputs);
	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override;

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

private:
	/** The actual Network pointer is not tracked so we need a WeakPtr too so we can check its validity on the game thread. */
	TWeakObjectPtr<UNeuralNetwork> StyleTransferNetworkWeakPtr;
	TObjectPtr<UNeuralNetwork> StyleTransferNetwork;

	FViewportClient* LinkedViewportClient;

	TSharedRef<int32, ESPMode::ThreadSafe> InferenceContext = MakeShared<int32>(-1);

	bool bIsEnabled = true;

	void AddRescalingTextureCopy(FRDGBuilder& GraphBuilder, FRDGTexture& RDGSourceTexture, FScreenPassRenderTarget& DestinationRenderTarget);
	FRDGTexture* TensorToTexture(FRDGBuilder& GraphBuilder, const FRDGTextureDesc& BaseDestinationDesc, const FNeuralTensor& SourceTensor);
	void TextureToTensor(FRDGBuilder& GraphBuilder, const FScreenPassTexture& SourceTexture, const FNeuralTensor& DestinationTensor);
};
