// Copyright Epic Games, Inc. All Rights Reserved.

using System.IO;
using UnrealBuildTool;

public class StyleTransfer : ModuleRules
{
	public StyleTransfer(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[]
			{
			}
		);

		string EngineDir = Path.GetFullPath(Target.RelativeEnginePath);
		PrivateIncludePaths.AddRange(
			new string[]
			{
				Path.Combine(EngineDir, "Source/Runtime/Renderer/Private"),
			}
		);


		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"NeuralNetworkInference",
			}
		);


		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"RHI",
				"RenderCore",
				"Renderer",
				"Projects",
				"StyleTransferShaders",
				"PixWinPlugin",
				"InputDevice",
			}
		);


		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
			}
		);
	}
}