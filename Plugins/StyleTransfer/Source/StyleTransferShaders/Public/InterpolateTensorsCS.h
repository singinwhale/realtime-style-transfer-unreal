// Copyright Manuel Wagner All Rights Reserved.

#pragma once

// GPU/RHI/shaders
#include "GlobalShader.h"
#include "RHI.h"
#include "ProfilingDebugging/RealtimeGPUProfiler.h"
#include "RenderGraphUtils.h"
#include "ShaderParameterUtils.h"



class STYLETRANSFERSHADERS_API FInterpolateTensorsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FInterpolateTensorsCS);
	SHADER_USE_PARAMETER_STRUCT(FInterpolateTensorsCS, FGlobalShader)


	static const FIntVector ThreadGroupSize;


	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float>, OutputUAV)
		// Unreal sadly does not support arrays of Buffers in shader compiler
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, InputSrvA)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, InputSrvB)
		SHADER_PARAMETER(float, Alpha)
		SHADER_PARAMETER(uint32, TensorVolume)
	END_SHADER_PARAMETER_STRUCT()

	// - FShader
	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
		return GetMaxSupportedFeatureLevel(Parameters.Platform) >= ERHIFeatureLevel::SM5;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment);
	// --

private:
};
