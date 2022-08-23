#include "StyleTransferShaders.h"

#include "Interfaces/IPluginManager.h"

DEFINE_LOG_CATEGORY(LogStyleTransferShaders);

#define LOCTEXT_NAMESPACE "FStyleTransferShadersModule"

void FStyleTransferShadersModule::StartupModule()
{
	// Add shader directory
	const TSharedPtr<IPlugin> StyleTransferPlugin = IPluginManager::Get().FindPlugin(TEXT("StyleTransfer"));
	if (StyleTransferPlugin.IsValid())
	{
		const FString RealShaderDirectory = StyleTransferPlugin->GetBaseDir() / TEXT("Shaders"); // TEXT("../../../Plugins/StyleTransfer/Shaders");
		const FString VirtualShaderDirectory = TEXT("/Plugins/StyleTransfer/Shaders");
		AddShaderSourceDirectoryMapping(VirtualShaderDirectory, RealShaderDirectory);
	}
	else
	{
		UE_LOG(LogStyleTransferShaders, Warning,
			TEXT("FStyleTransferModule::StartupModule(): StyleTransferPlugin was nullptr, shaders directory not added."));
	}
}

void FStyleTransferShadersModule::ShutdownModule()
{
    
}

#undef LOCTEXT_NAMESPACE
    
IMPLEMENT_MODULE(FStyleTransferShadersModule, StyleTransferShaders)