//
//  ClearVoiceCoreML.swift
//  ClearVoiceCoreML
//
//  CoreML-based audio processing providers
//

import Foundation
import CoreML
import ClearVoice
import ClearVoiceCore

// Re-export core types
@_exported import ClearVoice
@_exported import ClearVoiceCore

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

// MARK: - ClearVoice Extension

public extension ClearVoice {
    
    /// Configure with MossFormer GAN SE 16K (CoreML)
    /// - Parameters:
    ///   - provider: MossFormerGANCoreMLProvider instance
    ///   - model: Enhancement model identifier (default: .mossformerGAN)
    func configure(with provider: MossFormerGANCoreMLProvider, for model: EnhancementModel = .mossformerGAN) async throws {
        try await provider.load()
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
