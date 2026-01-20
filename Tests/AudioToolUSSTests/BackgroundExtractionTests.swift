//
//  BackgroundExtractionTests.swift
//  ClearVoiceTests
//
//  Test background extraction for GAN SE, USS, and Demucs
//

import XCTest
import ClearVoiceUSS
import ClearVoiceCoreML
import ClearVoiceMLX
import ClearVoiceCore
import AudioUtils
import MLX

final class BackgroundExtractionTests: XCTestCase {
    
    // Compute project root from source file path
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() } // file → ClearVoiceUSSTests → Tests → ClearVoice → ProjectTwo
        return url.path
    }()
    
    let outputDir = "\(projectRoot)/ClearVoice/Tests/ClearVoiceUSSTests/Fixtures"
    
    /// Test USS speech separation with background extraction
    func testUSSBackground() async throws {
        print("\n=== USS Background Extraction Test ===")
        
        // Load test audio at 32kHz (USS native rate)
        let testPath = "\(Self.projectRoot)/Models/uss_mlx_swift/USSSwift/Samples/harry_potter_short.wav"
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 32000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        print("Input: \(samples.count) samples @ 32kHz (\(String(format: "%.1f", Double(samples.count) / 32000))s)")
        
        // Load USS
        let uss = USSProviders.speechSeparation()
        try await uss.load()
        
        // Separate with background
        let start = Date()
        let input = AudioBuffer(samples: samples, sampleRate: 32000)
        let result = try await uss.separateWithBackground(input)
        let elapsed = Date().timeIntervalSince(start)
        
        print("Separation: \(String(format: "%.2f", elapsed))s")
        print("  Speech: \(result.separated.samples.count) samples, max: \(String(format: "%.4f", result.separated.samples.max() ?? 0))")
        print("  Background: \(result.background.samples.count) samples, max: \(String(format: "%.4f", result.background.samples.max() ?? 0))")
        
        // Save outputs
        try AudioSaver.saveWAV(MLXArray(result.separated.samples), to: "\(outputDir)/uss_speech.wav", sampleRate: 32000)
        try AudioSaver.saveWAV(MLXArray(result.background.samples), to: "\(outputDir)/uss_background.wav", sampleRate: 32000)
        print("✓ Saved: uss_speech.wav, uss_background.wav")
        
        XCTAssertGreaterThan(result.separated.samples.max() ?? 0, 0.01)
    }
    
    /// Test GAN SE CoreML with background extraction
    func testGANSEBackground() async throws {
        print("\n=== GAN SE CoreML Background Extraction Test ===")
        
        // Load test audio at 16kHz (GAN SE native rate)
        let testPath = "\(Self.projectRoot)/Models/mossformer_gan_se_coreml/test.wav"
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        print("Input: \(samples.count) samples @ 16kHz (\(String(format: "%.1f", Double(samples.count) / 16000))s), max: \(String(format: "%.4f", samples.max() ?? 0))")
        
        // Load reference output for comparison
        let refPath = "\(Self.projectRoot)/Models/mossformer_gan_se_coreml/enhanced_output_no_chunk.wav"
        let refAudio = try loader.loadMono(from: URL(fileURLWithPath: refPath))
        eval(refAudio)
        let refSamples = refAudio.asArray(Float.self)
        print("Reference: \(refSamples.count) samples, max: \(String(format: "%.4f", refSamples.max() ?? 0))")
        
        // Load GAN SE (auto-compiles .mlpackage)
        let modelPath = "\(Self.projectRoot)/Models/mossformer_gan_se_coreml/MossFormerGAN_256frames.mlpackage"
        let gan = MossFormerGANCoreMLProvider(modelPath: modelPath)
        try await gan.load()
        
        // Test standard process() first
        let input = AudioBuffer(samples: samples, sampleRate: 16000)
        let startStd = Date()
        let enhancedStd = try await gan.process(input)
        let elapsedStd = Date().timeIntervalSince(startStd)
        print("\nStandard process():")
        print("  Time: \(String(format: "%.2f", elapsedStd))s")
        print("  Output: \(enhancedStd.samples.count) samples, max: \(String(format: "%.4f", enhancedStd.samples.max() ?? 0))")
        
        // Compare with reference
        let minLen = min(enhancedStd.samples.count, refSamples.count)
        var diff: Float = 0
        for i in 0..<minLen {
            diff += abs(enhancedStd.samples[i] - refSamples[i])
        }
        let avgDiff = diff / Float(minLen)
        print("  Avg diff from reference: \(String(format: "%.6f", avgDiff))")
        
        // Save standard enhanced for inspection
        try AudioSaver.saveWAV(MLXArray(enhancedStd.samples), to: "\(outputDir)/ganse_enhanced_std.wav", sampleRate: 16000)
        
        // Process with background
        let start = Date()
        let result = try await gan.processWithBackground(input)
        let elapsed = Date().timeIntervalSince(start)
        
        print("\nprocessWithBackground():")
        print("  Time: \(String(format: "%.2f", elapsed))s")
        print("  Enhanced: \(result.enhanced.samples.count) samples, max: \(String(format: "%.4f", result.enhanced.samples.max() ?? 0))")
        print("  Background: \(result.background.samples.count) samples, max: \(String(format: "%.4f", result.background.samples.max() ?? 0))")
        
        // Save outputs
        try AudioSaver.saveWAV(MLXArray(result.enhanced.samples), to: "\(outputDir)/ganse_enhanced.wav", sampleRate: 16000)
        try AudioSaver.saveWAV(MLXArray(result.background.samples), to: "\(outputDir)/ganse_background.wav", sampleRate: 16000)
        print("✓ Saved: ganse_enhanced.wav, ganse_background.wav, ganse_enhanced_std.wav")
        
        XCTAssertGreaterThan(result.enhanced.samples.max() ?? 0, 0.01)
    }
    
    /// Test Demucs vocals/accompaniment separation
    func testDemucsBackground() async throws {
        print("\n=== Demucs Vocals/Accompaniment Test ===")
        
        // Load test audio at 44.1kHz (Demucs native rate)
        let testPath = "\(Self.projectRoot)/Models/demucs_mlx_swift/test.wav"
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 44100,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        print("Input: \(samples.count) samples @ 44.1kHz (\(String(format: "%.1f", Double(samples.count) / 44100))s)")
        
        // Load Demucs (all 4 source models)
        let weightsDir = "\(Self.projectRoot)/Models/demucs_mlx_swift/Weights"
        let demucs = DemucsProvider(weightsDirectory: weightsDir)
        try await demucs.loadAll()
        
        // Separate vocals and accompaniment
        let start = Date()
        let input = AudioBuffer(samples: samples, sampleRate: 44100)
        let result = try await demucs.separateVocalsWithAccompaniment(input)
        let elapsed = Date().timeIntervalSince(start)
        
        print("Separation: \(String(format: "%.2f", elapsed))s")
        print("  Vocals: \(result.vocals.samples.count) samples, max: \(String(format: "%.4f", result.vocals.samples.max() ?? 0))")
        print("  Accompaniment: \(result.accompaniment.samples.count) samples, max: \(String(format: "%.4f", result.accompaniment.samples.max() ?? 0))")
        
        // Save outputs
        try AudioSaver.saveWAV(MLXArray(result.vocals.samples), to: "\(outputDir)/demucs_vocals.wav", sampleRate: 44100)
        try AudioSaver.saveWAV(MLXArray(result.accompaniment.samples), to: "\(outputDir)/demucs_accompaniment.wav", sampleRate: 44100)
        print("✓ Saved: demucs_vocals.wav, demucs_accompaniment.wav")
        
        XCTAssertGreaterThan(result.vocals.samples.max() ?? 0, 0.001)
    }
    
    /// Test FRCRN SE with background extraction
    func testFRCRNBackground() async throws {
        print("\n=== FRCRN SE Background Extraction Test ===")
        
        // Load test audio at 16kHz (FRCRN native rate) - use same file as GAN SE test
        let testPath = "\(Self.projectRoot)/Models/mossformer_gan_se_coreml/test.wav"
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        print("Input: \(samples.count) samples @ 16kHz (\(String(format: "%.1f", Double(samples.count) / 16000))s), max: \(String(format: "%.4f", samples.max() ?? 0))")
        
        // Load FRCRN
        let weightsPath = "\(Self.projectRoot)/Models/frcrn_se_mlx_swift/Weights/frcrn_se_16k.safetensors"
        let frcrn = MLXProviders.frcrnSE16K(weightsPath: weightsPath)
        try await frcrn.load()
        
        // Process with background extraction
        let start = Date()
        let input = AudioBuffer(samples: samples, sampleRate: 16000)
        let result = try await frcrn.processWithBackground(input)
        let elapsed = Date().timeIntervalSince(start)
        
        print("Enhancement: \(String(format: "%.2f", elapsed))s")
        print("  Enhanced: \(result.enhanced.samples.count) samples, max: \(String(format: "%.4f", result.enhanced.samples.max() ?? 0))")
        print("  Background: \(result.background.samples.count) samples, max: \(String(format: "%.4f", result.background.samples.max() ?? 0))")
        
        // Save outputs
        try AudioSaver.saveWAV(MLXArray(result.enhanced.samples), to: "\(outputDir)/frcrn_enhanced.wav", sampleRate: 16000)
        try AudioSaver.saveWAV(MLXArray(result.background.samples), to: "\(outputDir)/frcrn_background.wav", sampleRate: 16000)
        print("✓ Saved: frcrn_enhanced.wav, frcrn_background.wav")
        
        XCTAssertGreaterThan(result.enhanced.samples.max() ?? 0, 0.01)
        XCTAssertGreaterThan(result.background.samples.max() ?? 0, 0.001)
    }
}
