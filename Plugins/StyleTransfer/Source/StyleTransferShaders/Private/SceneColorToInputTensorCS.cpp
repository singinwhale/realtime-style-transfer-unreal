// Copyright Manuel Wagner All Rights Reserved.

#include "SceneColorToInputTensorCS.h"

const FIntVector FSceneColorToInputTensorCS::ThreadGroupSize{8, 8, 1};


void FSceneColorToInputTensorCS::ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), ThreadGroupSize.X);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), ThreadGroupSize.Y);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Z"), ThreadGroupSize.Z);
}

IMPLEMENT_GLOBAL_SHADER(FSceneColorToInputTensorCS,
                        "/Plugins/StyleTransfer/Shaders/Private/SceneColorToInputTensor.usf",
                        "SceneColorToInputTensorCS", SF_Compute); // Path defined in StyleTransferModule.cpp
