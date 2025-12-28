//
//  ClearVoiceMLX.swift
//  ClearVoiceMLX
//
//  Public exports and convenience factory for MLX providers
//

import Foundation
import ClearVoice
import ClearVoiceCore

// Re-export core types
@_exported import ClearVoice
@_exported import ClearVoiceCore

/// Factory for creating MLX-based providers
public struct MLXProviders {
    
    // MARK: - Speech Enhancement
    
    /// Create MossFormer2 SE 48K enhancer
    public static func mossformer2SE48K(
        weightsPath: String? = nil,
        precision: String = "fp32"
    ) -> MossFormer2SE48KProvider {
        MossFormer2SE48KProvider(weightsPath: weightsPath, precision: precision)
    }
    
    /// Create FRCRN SE 16K enhancer
    public static func frcrnSE16K(weightsPath: String) -> FRCRNSE16KProvider {
        FRCRNSE16KProvider(weightsPath: weightsPath)
    }
    
    // Note: MossFormer GAN SE 16K is now in ClearVoiceCoreML
    // Use CoreMLProviders.mossformerGANSE16K() instead
    
    // MARK: - Speaker Separation
    
    /// Create MossFormer2 SS provider for speaker separation
    /// - Parameters:
    ///   - model: Model variant (.twoSpeaker, .threeSpeaker, .twoSpeakerWHAMR)
    ///   - weightsPath: Path to model weights
    public static func mossformer2SS(
        model: MossFormer2SSProvider.Model,
        weightsPath: String
    ) -> MossFormer2SSProvider {
        MossFormer2SSProvider(model: model, weightsPath: weightsPath)
    }
    
    // MARK: - Source Separation
    
    /// Create Demucs source separator (vocals, drums, bass, other)
    public static func demucs(weightsDirectory: String) -> DemucsProvider {
        DemucsProvider(weightsDirectory: weightsDirectory)
    }
    
    // MARK: - Super Resolution
    
    /// Create MossFormer2 SR 48K upsampler
    public static func mossformer2SR48K(weightsPath: String, configPath: String) -> MossFormer2SR48KProvider {
        MossFormer2SR48KProvider(weightsPath: weightsPath, configPath: configPath)
    }
}

/// ClearVoice extension for registering MLX providers
extension ClearVoice {
    
    /// Configure with MLX enhancement provider
    public func configure(enhancer: MossFormer2SE48KProvider, for model: EnhancementModel = .mossformerSE48k) async throws {
        try await enhancer.load()
        self.register(enhancer: enhancer, for: model)
    }
    
    /// Configure with MLX FRCRN provider
    public func configure(enhancer: FRCRNSE16KProvider, for model: EnhancementModel = .frcrn) async throws {
        try await enhancer.load()
        self.register(enhancer: enhancer, for: model)
    }
    
    // Note: MossFormer GAN SE configure is now in ClearVoiceCoreML
    
    /// Configure with MLX speaker separator
    public func configure(separator: MossFormer2SSProvider, for model: SeparationModel = .mossformer2spk) async throws {
        try await separator.load()
        self.register(separator: separator, for: model)
    }
    
    /// Configure with Demucs source separator
    public func configure(separator: DemucsProvider, for model: SeparationModel = .demucs, sources: [DemucsProvider.Source] = DemucsProvider.Source.allCases) async throws {
        for source in sources {
            try separator.loadSync(source: source)
        }
        self.register(separator: separator, for: model)
    }
    
    /// Configure with super resolution provider
    public func configure(upsampler: MossFormer2SR48KProvider) async throws {
        try await upsampler.load()
        self.register(upscaler: upsampler)
    }
}
