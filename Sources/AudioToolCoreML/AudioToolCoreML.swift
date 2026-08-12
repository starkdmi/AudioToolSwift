//
//  AudioToolCoreML.swift
//  AudioToolCoreML
//
//  CoreML-based audio processing providers
//

import Foundation
import CoreML
import AudioTool
import AudioToolCore

// Re-export core types
@_exported import AudioTool
@_exported import AudioToolCore

// MARK: - CoreML Providers Factory

/// Factory for creating CoreML-based audio processing providers
public struct CoreMLProviders {
    
    /// Create MossFormer GAN SE 16K provider (CoreML)
    /// - Parameters:
    ///   - modelPath: Path to .mlpackage file
    ///   - computeUnits: CoreML compute units (default: cpuAndGPU)
    /// - Returns: MossFormerGANCoreMLProvider instance
    public static func mossformerGANSE16K(
        modelPath: String,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) -> MossFormerGANCoreMLProvider {
        MossFormerGANCoreMLProvider(modelPath: modelPath, computeUnits: computeUnits)
    }
}

// MARK: - AudioTool Extension

public extension AudioEngine {
    
    /// Configure with MossFormer GAN SE 16K (CoreML)
    /// - Parameters:
    ///   - provider: MossFormerGANCoreMLProvider instance
    ///   - model: Enhancement model identifier (default: .mossformerGAN)
    func configure(with provider: MossFormerGANCoreMLProvider, for model: EnhancementModel = .mossformerGAN) async throws {
        // A load the residency manager knows about, then registration.
        try await self.preload(provider)
        self.register(enhancer: provider, for: model)
    }
}

// MARK: - Compute Units Helper

public extension MLComputeUnits {
    /// GPU preferred for best performance
    static var gpuPreferred: MLComputeUnits { .cpuAndGPU }
    
    /// ANE (Neural Engine) for efficiency
    static var anePreferred: MLComputeUnits { .cpuAndNeuralEngine }
    
    /// All available compute units
    static var allAvailable: MLComputeUnits { .all }
}
