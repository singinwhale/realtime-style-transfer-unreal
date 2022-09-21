// Copyright Manuel Wagner All Rights Reserved.

#include "ShadowMaskToInputTensorCS.h"

const FIntVector FShadowMaskToInputTensorCS::ThreadGroupSize{8, 8, 1};

void FShadowMaskToInputTensorCS::ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), ThreadGroupSize.X);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), ThreadGroupSize.Y);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Z"), ThreadGroupSize.Z);
}


IMPLEMENT_GLOBAL_SHADER(FShadowMaskToInputTensorCS,
						"/Plugins/StyleTransfer/Shaders/Private/ShadowMaskToInputTensor.usf",
						"ShadowMaskToInputTensorCS", SF_Compute); // Path defined in StyleTransferModule.cpp
