// Copyright Epic Games, Inc. All Rights Reserved.

#include "SceneColorToInputTensorCS.h"
#include "Utils.h"



void FSceneColorToInputTensorCS::ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), THREADGROUP_SIZE_X);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), THREADGROUP_SIZE_Y);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Z"), 1);
}

IMPLEMENT_GLOBAL_SHADER(FSceneColorToInputTensorCS,
                        "/Plugins/StyleTransfer/Shaders/Private/SceneColorToInputTensor.usf",
                        "SceneColorToInputTensorCS", SF_Compute); // Path defined in StyleTransferModule.cpp
