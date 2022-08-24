// Copyright Epic Games, Inc. All Rights Reserved.

#include "OutputTensorToSceneColorCS.h"
#include "Utils.h"

const FIntVector FOutputTensorToSceneColorCS::ThreadGroupSize{8, 8, 1};

void FOutputTensorToSceneColorCS::ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), ThreadGroupSize.X);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), ThreadGroupSize.Y);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Z"), ThreadGroupSize.Z);
}

IMPLEMENT_GLOBAL_SHADER(FOutputTensorToSceneColorCS, "/Plugins/StyleTransfer/Shaders/Private/OutputTensorToSceneColor.usf", "OutputTensorToSceneColorCS", SF_Compute); // Path defined in StyleTransferModule.cpp