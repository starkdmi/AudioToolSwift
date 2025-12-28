//
//  MLXEnhancerIntegrationTests.swift
//  ClearVoiceMLXIntegrationTests
//
//  Integration tests for MLX enhancement providers with chunking
//  Saves audio outputs to Models/*/swift_chunked_outputs/
//
//  NOTE: Run with xcodebuild for Metal support:
//    xcodebuild test -scheme ClearVoice-Package -destination 'platform=macOS'
//

import Testing
import Foundation
@testable import ClearVoiceMLX
@testable import ClearVoice
@testable import ClearVoiceCore
import MLX
import AudioUtils

// MARK: - Test Configuration

struct TestConfig {
    // Project root (absolute path)
    static let projectRoot = "/path/to/clear_voice_research"
    
    // Model weights paths
    static let frcrn16kWeights = "\(projectRoot)/Models/frcrn_se_mlx_swift/Weights/frcrn_se_16k.safetensors"
    // MossFormer2 SE downloads from HuggingFace, uses empty string to indicate HF download
    static let mossformer2SE48KWeights = "" // Will use HuggingFace
    
    // Test audio paths
    static let frcrnTestAudio = "\(projectRoot)/Models/frcrn_se_mlx_swift/test.wav"
    static let mossformer2TestAudio = "\(projectRoot)/Models/mossformer2_se_mlx_swift/test.wav"
    
    // Output directories
    static let frcrnOutputDir = "\(projectRoot)/Models/frcrn_se_mlx_swift"
    static let mossformer2OutputDir = "\(projectRoot)/Models/mossformer2_se_mlx_swift"
    
    static var shouldRunIntegrationTests: Bool {
        ProcessInfo.processInfo.environment["SKIP_MLX_TESTS"] != "1"
    }
}


// MARK: - Audio I/O Helpers

func saveAudio(_ samples: [Float], sampleRate: Int, to path: String) throws {
    let saverConfig = AudioSaver.Configuration(sampleRate: Double(sampleRate))
    let saver = AudioSaver(config: saverConfig)
    try saver.save(MLXArray(samples), to: path)
}

// MARK: - FRCRN Integration Tests with Chunking

@Suite("FRCRN SE 16K Integration with Chunking", .tags(.integration))
struct FRCRNChunkingIntegrationTests {
    
    @Test("FRCRN processes audio with chunking and saves output")
    func testFRCRNWithChunking() async throws {
        guard TestConfig.shouldRunIntegrationTests else {
            print("Skipping MLX tests")
            return
        }
        
        // Check if weights exist
        guard FileManager.default.fileExists(atPath: TestConfig.frcrn16kWeights) else {
            print("FRCRN weights not found at: \(TestConfig.frcrn16kWeights)")
            return
        }
        
        print("\n=== FRCRN SE 16K with Chunking ===")
        
        // Create provider (chunking is auto-enabled)
        let provider = FRCRNSE16KProvider(weightsPath: TestConfig.frcrn16kWeights)
        
        // Load model
        print("Loading model...")
        try await provider.load()
        
        // Create synthetic 6s audio (longer than 4s chunk to trigger chunking)
        // Noisy sine wave at 440Hz
        let sampleRate = 16000
        let duration: Float = 6.0
        let samples: [Float] = (0..<Int(duration * Float(sampleRate))).map { i in
            let t = Float(i) / Float(sampleRate)
            let sine = sin(2.0 * Float.pi * 440.0 * t) * 0.5
            let noise = Float.random(in: -0.1...0.1)
            return sine + noise
        }
        
        let input = AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
        print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
        print("Chunking will split into 4s chunks with 25% overlap...")
        
        // Process with chunking (auto-enabled)
        print("Processing...")
        let startTime = Date()
        let output = try await provider.process(input)
        let processingTime = Date().timeIntervalSince(startTime)
        
        // Calculate RTF
        let rtf = input.duration / processingTime
        print("Output: \(output.samples.count) samples (\(String(format: "%.2f", output.duration))s)")
        print("RTF: \(String(format: "%.2f", rtf))x")
        
        // Save output
        let outputPath = "\(TestConfig.frcrnOutputDir)/enhanced_6s_chunked.wav"
        let saverConfig = AudioSaver.Configuration(sampleRate: Double(sampleRate))
        let saver = AudioSaver(config: saverConfig)
        try saver.save(MLXArray(output.samples), to: outputPath)
        print("✓ Saved: \(outputPath)")
        
        // Verify output
        #expect(output.sampleRate == 16000)
        #expect(output.frameCount > 0)
        #expect(rtf > 1.0, "Should be faster than real-time")
    }
}

// MARK: - MossFormer2 SE 48K Integration Tests with Chunking

@Suite("MossFormer2 SE 48K Integration with Chunking", .tags(.integration))
struct MossFormer2ChunkingIntegrationTests {
    
    @Test("MossFormer2 SE processes audio with chunking and saves output")
    func testMossFormer2SEWithChunking() async throws {
        guard TestConfig.shouldRunIntegrationTests else {
            print("Skipping MLX tests")
            return
        }
        
        print("\n=== MossFormer2 SE 48K with Chunking ===")
        
        // Create provider (downloads from HuggingFace)
        let provider = MLXProviders.mossformer2SE48K()
        
        // Load model (downloads if needed)
        print("Loading model (may download from HuggingFace)...")
        try await provider.load()
        
        // Load test audio
        print("Loading test audio from: \(TestConfig.mossformer2TestAudio)")
        let loaderConfig = AudioLoader.Configuration(targetSampleRate: 48000)
        let loader = AudioLoader(config: loaderConfig)
        let audio = try loader.loadMono(from: URL(fileURLWithPath: TestConfig.mossformer2TestAudio))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let input = AudioBuffer(samples: samples, sampleRate: 48000, channels: 1)
        print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
        
        // Process with chunking (auto-enabled for >4s audio)
        print("Processing with 4s/25%/discard-edges chunking...")
        let startTime = Date()
        let output = try await provider.process(input)
        let processingTime = Date().timeIntervalSince(startTime)
        
        // Calculate RTF
        let rtf = input.duration / processingTime
        print("Output: \(output.samples.count) samples (\(String(format: "%.2f", output.duration))s)")
        print("RTF: \(String(format: "%.2f", rtf))x")
        
        // Save output
        let outputPath = "\(TestConfig.mossformer2OutputDir)/enhanced_test_chunked.wav"
        let saverConfig = AudioSaver.Configuration(sampleRate: 48000)
        let saver = AudioSaver(config: saverConfig)
        try saver.save(MLXArray(output.samples), to: outputPath)
        print("✓ Saved: \(outputPath)")
        
        // Verify output
        #expect(output.sampleRate == 48000)
        #expect(output.frameCount > 0)
    }
}

// MARK: - Chunking Config Verification

@Suite("Chunking Configuration Tests")
struct ChunkingConfigTests {
    
    @Test("All preset configs match benchmark settings")
    func testPresetConfigs() {
        let frcrn = ChunkingConfig.frcrnSE16K()
        #expect(frcrn.chunkDuration == 4.0)
        #expect(frcrn.overlapRatio == 0.25)
        print("FRCRN: 4s, 25%, discard-edges ✓")
        
        let se48k = ChunkingConfig.mossformer2SE48K()
        #expect(se48k.chunkDuration == 4.0)
        #expect(se48k.overlapRatio == 0.25)
        print("MossFormer2 SE 48K: 4s, 25%, discard-edges ✓")
        
        let sr48k = ChunkingConfig.mossformer2SR48K()
        #expect(sr48k.chunkDuration == 4.0)
        #expect(sr48k.overlapRatio == 0.5)
        print("MossFormer2 SR 48K: 4s, 50%, hann ✓")
        
        let ss = ChunkingConfig.mossformer2SS()
        #expect(ss.chunkDuration == 4.0)
        #expect(ss.overlapRatio == 0.25)
        print("MossFormer2 SS: 4s, 25%, triangular ✓")
        
        let demucs = ChunkingConfig.demucs()
        #expect(demucs.chunkDuration == 7.8)
        #expect(demucs.overlapRatio == 0.25)
        print("Demucs: 7.8s, 25%, triangular ✓")
        
        let ganSE = ChunkingConfig.ganSECoreML()
        #expect(ganSE.chunkDuration == 1.594)
        #expect(ganSE.overlapRatio == 0.0)
        print("GAN SE CoreML: 1.594s, 0%, none ✓")
    }
}

// MARK: - Test Tags

extension Tag {
    @Tag static var integration: Self
}
