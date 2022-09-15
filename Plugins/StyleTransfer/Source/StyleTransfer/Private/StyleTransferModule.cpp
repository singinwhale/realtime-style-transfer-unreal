// Copyright Manuel Wagner All Rights Reserved.

#include "StyleTransferModule.h"

#include "ShaderCore.h"
#include "Interfaces/IPluginManager.h"
#include "Logging/LogMacros.h"

DEFINE_LOG_CATEGORY(LogStyleTransfer)

#define LOCTEXT_NAMESPACE "FStyleTransferModule"

void FStyleTransferModule::StartupModule()
{
}

void FStyleTransferModule::ShutdownModule()
{
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FStyleTransferModule, StyleTransfer)
