// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

// GPU/RHI/shaders
#include "GlobalShader.h"
#include "RHI.h"
#include "ProfilingDebugging/RealtimeGPUProfiler.h"
#include "RenderGraphUtils.h"
#include "ShaderParameterUtils.h"



class STYLETRANSFERSHADERS_API FSceneColorToInputTensorCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSceneColorToInputTensorCS);
	SHADER_USE_PARAMETER_STRUCT(FSceneColorToInputTensorCS, FGlobalShader)

	static const uint32 THREADGROUP_SIZE_X = 8;
	static const uint32 THREADGROUP_SIZE_Y = 8;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Input variables
		SHADER_PARAMETER(uint32, TensorVolume)
		// SRV/UAV variables
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float>, OutputUAV)
		// Optional SRV/UAV variables
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputTexture)
	END_SHADER_PARAMETER_STRUCT()

	// - FShader
	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
		return GetMaxSupportedFeatureLevel(Parameters.Platform) >= ERHIFeatureLevel::SM5;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment);
	// --

private:
};
