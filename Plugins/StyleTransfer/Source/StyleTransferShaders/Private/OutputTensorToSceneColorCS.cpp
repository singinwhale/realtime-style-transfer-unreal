// Copyright Epic Games, Inc. All Rights Reserved.

#include "OutputTensorToSceneColorCS.h"
#include "Utils.h"



void FOutputTensorToSceneColorCS::ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
}

IMPLEMENT_GLOBAL_SHADER(FOutputTensorToSceneColorCS, "/Plugins/StyleTransfer/Shaders/Private/OutputTensorToSceneColor.usf", "OutputTensorToSceneColorCS", SF_Compute); // Path defined in StyleTransferModule.cpp