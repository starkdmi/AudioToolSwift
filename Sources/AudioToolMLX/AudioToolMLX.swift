//
//  AudioToolMLX.swift
//  AudioToolMLX
//
//  Public exports and convenience factory for MLX providers
//

import Foundation
import AudioTool
import AudioToolCore

// Re-export core types
@_exported import AudioTool
@_exported import AudioToolCore

/// Factory for creating MLX-based providers
public struct MLXProviders {
    
    // MARK: - Speech Enhancement
    
    /// Create MossFormer2 SE 48K enhancer with precision selection
    /// - Parameter precision: Model precision (.fp32 or .fp16)
    public static func mossformer2SE48K(
        precision: ModelPrecision = .fp32
    ) -> MossFormer2SE48KProvider {
        MossFormer2SE48KProvider(precision: precision)
    }
    
    /// Create MossFormer2 SE 48K enhancer with explicit weights path
    public static func mossformer2SE48K(weightsPath: String) -> MossFormer2SE48KProvider {
        MossFormer2SE48KProvider(weightsPath: weightsPath)
    }
    
    /// Create FRCRN SE 16K enhancer
    public static func frcrnSE16K(weightsPath: String) -> FRCRNSE16KProvider {
        FRCRNSE16KProvider(weightsPath: weightsPath)
    }
    
    // Note: MossFormer GAN SE 16K is now in AudioToolCoreML
    // Use CoreMLProviders.mossformerGANSE16K() instead
    
    // MARK: - Speaker Separation
    
    /// Create MossFormer2 SS provider with precision selection
    /// - Parameters:
    ///   - model: Model variant (.twoSpeaker, .threeSpeaker, .twoSpeakerWHAMR)
    ///   - precision: Model precision (currently .fp32 only)
    public static func mossformer2SS(
        model: MossFormer2SSProvider.Model,
        precision: ModelPrecision = .fp32
    ) -> MossFormer2SSProvider {
        MossFormer2SSProvider(model: model, precision: precision)
    }
    
    /// Create MossFormer2 SS provider with explicit weights path
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
    
    /// Create MossFormer2 SR 48K upsampler with precision selection
    /// - Parameter precision: Model precision (currently .fp32 only)
    public static func mossformer2SR48K(precision: ModelPrecision = .fp32) -> MossFormer2SR48KProvider {
        MossFormer2SR48KProvider(precision: precision)
    }
    
    /// Create MossFormer2 SR 48K upsampler with explicit paths
    public static func mossformer2SR48K(weightsPath: String, configPath: String) -> MossFormer2SR48KProvider {
        MossFormer2SR48KProvider(weightsPath: weightsPath, configPath: configPath)
    }
}

/// AudioEngine extension for registering MLX providers
///
/// Each of these loads through ``AudioEngine/preload(_:)`` rather than calling the
/// provider's `load()` directly: `preload` goes through the residency manager, so a
/// configured model counts against the memory budget instead of sitting alongside
/// it, and two concurrent configurations of the same provider share one load.
///
/// Registration comes after the load, so a `configure` that fails leaves the engine
/// exactly as it was. `preload` takes the provider directly and never consults the
/// registry, so nothing is gained by registering first - and registering first meant
/// a failed reconfiguration replaced a working provider with the one that had just
/// failed to load.
extension AudioEngine {

    /// Configure with MLX enhancement provider
    public func configure(enhancer: MossFormer2SE48KProvider, for model: EnhancementModel = .mossformerSE48k) async throws {
        try await self.preload(enhancer)
        self.register(enhancer: enhancer, for: model)
    }

    /// Configure with MLX FRCRN provider
    public func configure(enhancer: FRCRNSE16KProvider, for model: EnhancementModel = .frcrn) async throws {
        try await self.preload(enhancer)
        self.register(enhancer: enhancer, for: model)
    }

    // Note: MossFormer GAN SE configure is now in AudioToolCoreML

    /// Configure with MLX speaker separator
    public func configure(separator: MossFormer2SSProvider, for model: SeparationModel = .mossformer2spk) async throws {
        try await self.preload(separator)
        self.register(separator: separator, for: model)
    }
    
    /// Configure with the Demucs music separator.
    ///
    /// Not registered into the speech-separator registry: stems are named parts of a
    /// mix, not interchangeable speaker slots, so Demucs conforms to `MusicSeparator`
    /// and is used directly. Each stem has its own weight file, so only the requested
    /// ones are loaded.
    public func configure(musicSeparator: DemucsProvider, stems: [DemucsProvider.Stem] = DemucsProvider.Stem.allCases) async throws {
        for stem in stems {
            try await musicSeparator.load(stem: stem)
        }
    }
    
    /// Configure with super resolution provider
    public func configure(upsampler: MossFormer2SR48KProvider) async throws {
        try await self.preload(upsampler)
        self.register(upscaler: upsampler)
    }
}
