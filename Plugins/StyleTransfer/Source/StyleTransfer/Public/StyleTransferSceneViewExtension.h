#pragma once
#include "SceneViewExtension.h"

class UNeuralNetwork;

class FStyleTransferSceneViewExtension : public FSceneViewExtensionBase
{
public:
	using Ptr = TSharedPtr<FStyleTransferSceneViewExtension, ESPMode::ThreadSafe>;
	using Ref = TSharedRef<FStyleTransferSceneViewExtension, ESPMode::ThreadSafe>;

	FStyleTransferSceneViewExtension(const FAutoRegister& AutoRegister, FViewportClient* AssociatedViewportClient, UNeuralNetwork* InStyleTransferNetwork);

	// - ISceneViewExtension
	virtual void SubscribeToPostProcessingPass(EPostProcessingPass Pass, FAfterPassCallbackDelegateArray& InOutPassCallbacks, bool bIsPassEnabled) override;
	FScreenPassTexture PostProcessPassAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View,
	                                                            const FPostProcessMaterialInputs& InOutInputs);
	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override;
	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override {};
	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override {};
	// --

private:
	TObjectPtr<UNeuralNetwork> StyleTransferNetwork;

	FViewportClient* LinkedViewportClient;

	int32 InferenceContext = -1;
};
