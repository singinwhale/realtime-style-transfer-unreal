// Copyright Manuel Wagner All Rights Reserved.

#include "InterpolateTensorsCS.h"

const FIntVector FInterpolateTensorsCS::ThreadGroupSize{64, 1, 1};


void FInterpolateTensorsCS::ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);

	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_X"), ThreadGroupSize.X);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Y"), ThreadGroupSize.Y);
	OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE_Z"), ThreadGroupSize.Z);
}

IMPLEMENT_GLOBAL_SHADER(FInterpolateTensorsCS,
                        "/Plugins/StyleTransfer/Shaders/Private/InterpolateTensors.usf",
                        "InterpolateTensorsCS", SF_Compute); // Path defined in StyleTransferModule.cpp
