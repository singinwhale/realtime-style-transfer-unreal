// Copyright Manuel Wagner All Rights Reserved.

#pragma once

// GPU/RHI/shaders
#include "GlobalShader.h"
#include "RHI.h"
#include "ProfilingDebugging/RealtimeGPUProfiler.h"
#include "RenderGraphUtils.h"
#include "ShaderParameterUtils.h"


class STYLETRANSFERSHADERS_API FShadowMaskToInputTensorCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FShadowMaskToInputTensorCS);
	SHADER_USE_PARAMETER_STRUCT(FShadowMaskToInputTensorCS, FGlobalShader)


	static const FIntVector ThreadGroupSize;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Input variables
		SHADER_PARAMETER(uint32, TensorVolume)
		// SRV/UAV variables
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float>, OutputUAV)
		// Optional SRV/UAV variables
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, InputTextureSampler)
		SHADER_PARAMETER(FIntPoint, OutputDimensions)
		SHADER_PARAMETER(FVector2f, HalfPixelUV)
	END_SHADER_PARAMETER_STRUCT()

	// - FShader
	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
		return GetMaxSupportedFeatureLevel(Parameters.Platform) >= ERHIFeatureLevel::SM5;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment);
	// --

private:
};
