// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

// GPU/RHI/shaders
#include "GlobalShader.h"
#include "RHI.h"
#include "ProfilingDebugging/RealtimeGPUProfiler.h"
#include "RenderGraphUtils.h"
#include "ShaderParameterUtils.h"



class STYLETRANSFERSHADERS_API FOutputTensorToSceneColorCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FOutputTensorToSceneColorCS);
	SHADER_USE_PARAMETER_STRUCT(FOutputTensorToSceneColorCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(uint32, TensorVolume)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, InputTensor)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D, OutputTexture)
	END_SHADER_PARAMETER_STRUCT()

	// - FShader
	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
		return GetMaxSupportedFeatureLevel(Parameters.Platform) >= ERHIFeatureLevel::SM5;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment);
	// --

private:
};
