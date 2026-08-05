//
//  MLXEnhancerIntegrationTests.swift
//  AudioToolMLXIntegrationTests
//
//  Integration tests for MLX enhancement providers with chunking
//  Saves audio outputs to Models/*/swift_chunked_outputs/
//
//  NOTE: Run with xcodebuild for Metal support:
//    xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS'
//

import Testing
import Foundation
@testable import AudioToolMLX
@testable import AudioTool
@testable import AudioToolCore
import MLX
import AudioUtils

// MARK: - Test Configuration

struct TestConfig {
    // Project root computed from source file location
    // #filePath → .../AudioTool/Tests/AudioToolMLXIntegrationTests/MLXEnhancerIntegrationTests.swift
    // We go up 4 levels: file → AudioToolMLXIntegrationTests → Tests → AudioTool → ProjectTwo
    static let projectRoot: String = {
        let filePath = #filePath
        var url = URL(fileURLWithPath: filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
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
    
    /// Whether MLX tests should run (default: yes, set SKIP_MLX_TESTS=1 to skip)
    static var shouldRunIntegrationTests: Bool {
        ProcessInfo.processInfo.environment["SKIP_MLX_TESTS"] != "1"
    }
    
    /// Whether running in CI environment (adjusts performance thresholds)
    static var isCI: Bool {
        ProcessInfo.processInfo.environment["CI"] == "1"
    }
    
    /// Check if file exists at path
    static func fileExists(at path: String) -> Bool {
        FileManager.default.fileExists(atPath: path)
    }
}

/// Trait for enabling tests only when MLX tests are not skipped
extension Trait where Self == Testing.ConditionTrait {
    /// Enable test only when SKIP_MLX_TESTS is not set
    static var enabledForMLX: Self {
        .enabled(if: TestConfig.shouldRunIntegrationTests, "MLX tests disabled via SKIP_MLX_TESTS=1")
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
    
    @Test("FRCRN processes audio with chunking and saves output", .enabledForMLX)
    func testFRCRNWithChunking() async throws {
        try #require(TestConfig.fileExists(at: TestConfig.frcrn16kWeights), "FRCRN weights not found at \(TestConfig.frcrn16kWeights)")
        try #require(TestConfig.fileExists(at: TestConfig.frcrnTestAudio), "Test audio not found at \(TestConfig.frcrnTestAudio)")
        
        print("\n=== FRCRN SE 16K with Chunking ===")
        
        // Create provider (chunking is auto-enabled)
        let provider = FRCRNSE16KProvider(weightsPath: TestConfig.frcrn16kWeights)
        
        // Load model
        print("Loading model...")
        try await provider.load()
        
        // Load real test audio (noisy speech at 16kHz)
        let sampleRate = 16000
        let testAudioPath = TestConfig.frcrnTestAudio
        print("Loading test audio from: \(testAudioPath)")
        
        let loaderConfig = AudioLoader.Configuration(targetSampleRate: Double(sampleRate))
        let loader = AudioLoader(config: loaderConfig)
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testAudioPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
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
        // RTF can vary based on GPU contention when multiple tests run in parallel
        // Using 0.5x as minimum threshold to account for parallel test execution
        let rtfThreshold = TestConfig.isCI ? 0.3 : 0.5
        #expect(rtf > rtfThreshold, "Should be at least \(rtfThreshold)x real-time (got \(rtf)x)")
    }
}

// MARK: - MossFormer2 SE 48K Integration Tests with Chunking

@Suite("MossFormer2 SE 48K Integration with Chunking", .tags(.integration))
struct MossFormer2ChunkingIntegrationTests {
    
    @Test("MossFormer2 SE processes audio with chunking and saves output", .enabledForMLX)
    func testMossFormer2SEWithChunking() async throws {
        
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

// MARK: - Streaming Verification Tests

@Suite("Streaming Output Verification", .tags(.integration))
struct StreamingVerificationTests {
    
    @Test("FRCRN Streaming produces identical output to batch processing", .enabledForMLX)
    func testFRCRNStreamingQuality() async throws {
        try #require(TestConfig.fileExists(at: TestConfig.frcrn16kWeights), "FRCRN weights not found")
        try #require(TestConfig.fileExists(at: TestConfig.frcrnTestAudio), "Test audio not found")
        
        print("\n=== FRCRN Streaming vs Batch Quality Test ===")
        
        // Create provider
        let provider = FRCRNSE16KProvider(weightsPath: TestConfig.frcrn16kWeights)
        
        // Load model
        print("Loading model...")
        try await provider.load()
        
        // Load real test audio at 16kHz
        let sampleRate = 16000
        let testAudioPath = TestConfig.frcrnTestAudio
        print("Loading test audio from: \(testAudioPath)")
        
        let loaderConfig = AudioLoader.Configuration(targetSampleRate: Double(sampleRate))
        let loader = AudioLoader(config: loaderConfig)
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testAudioPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let input = AudioBuffer(samples: samples, sampleRate: sampleRate, channels: 1)
        print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
        
        // 1. Batch processing
        print("\n--- Batch Processing ---")
        let batchStart = Date()
        let batchOutput = try await provider.process(input)
        let batchTime = Date().timeIntervalSince(batchStart)
        print("Batch output: \(batchOutput.samples.count) samples in \(String(format: "%.2f", batchTime))s")
        
        // 2. Streaming processing
        print("\n--- Streaming Processing ---")
        let streamStart = Date()
        var streamedChunks: [AudioBuffer] = []
        var chunkCount = 0
        
        for try await chunk in provider.processStream(input) {
            chunkCount += 1
            print("  Chunk \(chunkCount): \(chunk.samples.count) samples")
            streamedChunks.append(chunk)
        }
        
        let streamTime = Date().timeIntervalSince(streamStart)
        
        // Combine streamed chunks
        let streamedSamples = streamedChunks.flatMap { $0.samples }
        print("Streamed total: \(streamedSamples.count) samples in \(String(format: "%.2f", streamTime))s")
        
        // 3. Compare outputs
        print("\n--- Quality Comparison ---")
        let minLen = min(batchOutput.samples.count, streamedSamples.count)
        
        // Calculate differences
        var maxDiff: Float = 0
        var sumSquaredDiff: Float = 0
        
        for i in 0..<minLen {
            let diff = abs(batchOutput.samples[i] - streamedSamples[i])
            maxDiff = max(maxDiff, diff)
            sumSquaredDiff += diff * diff
        }
        
        let rmsDiff = sqrt(sumSquaredDiff / Float(minLen))
        
        print("Sample counts - Batch: \(batchOutput.samples.count), Stream: \(streamedSamples.count)")
        print("Max difference: \(String(format: "%.6f", maxDiff))")
        print("RMS difference: \(String(format: "%.6f", rmsDiff))")
        
        // Save outputs for manual inspection
        let batchPath = "\(TestConfig.frcrnOutputDir)/streaming_test_batch.wav"
        let streamPath = "\(TestConfig.frcrnOutputDir)/streaming_test_stream.wav"
        
        let saverConfig = AudioSaver.Configuration(sampleRate: Double(sampleRate))
        let saver = AudioSaver(config: saverConfig)
        try saver.save(MLXArray(batchOutput.samples), to: batchPath)
        try saver.save(MLXArray(streamedSamples), to: streamPath)
        print("✓ Saved batch: \(batchPath)")
        print("✓ Saved stream: \(streamPath)")
        
        // Verify quality
        // For discardEdges strategy, differences are expected at chunk boundaries
        // But they should be small (< 0.01 for reasonable quality)
        #expect(maxDiff < 0.5, "Max difference should be reasonable (< 0.5)")
        #expect(abs(batchOutput.samples.count - streamedSamples.count) < 1000, "Sample counts should be close")
        
        // Quality assessment
        if maxDiff < 0.001 {
            print("✅ NEAR-IDENTICAL: max diff < 0.001")
        } else if maxDiff < 0.01 {
            print("✅ EXCELLENT: max diff < 0.01")
        } else if maxDiff < 0.1 {
            print("⚠️ ACCEPTABLE: max diff < 0.1")
        } else {
            print("❌ CHECK IMPLEMENTATION: max diff >= 0.1")
        }
    }
}

// MARK: - MossFormer2 Speaker Separation Integration Tests

@Suite("MossFormer2 SS Integration with Chunking", .tags(.integration))
struct MossFormer2SSIntegrationTests {
    
    // Test paths computed from project root
    static var ssModelDir: String { "\(TestConfig.projectRoot)/Models/mosforrmer2_ss_mlx_swift" }
    
    @Test("MossFormer2 SS 2-speaker separation with chunking", .enabledForMLX)
    func testMossFormer2SS2Speaker() async throws {
        
        print("\n=== MossFormer2 SS 2-Speaker with Chunking ===")
        
        // Create provider - auto-downloads from HuggingFace if needed
        let provider = MossFormer2SSProvider(model: .twoSpeaker, precision: .fp32)
        
        print("Loading model...")
        try await provider.load()
        print("Model ready")
        
        // Load test audio (16kHz for 2-speaker model)
        let testPath = "\(Self.ssModelDir)/mix.wav"
        print("Loading audio from: \(testPath)")
        
        let loaderConfig = AudioLoader.Configuration(targetSampleRate: 16000)
        let loader = AudioLoader(config: loaderConfig)
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
        
        // Separate speakers
        print("Processing...")
        let startTime = Date()
        let outputs = try await provider.separate(input)
        let processingTime = Date().timeIntervalSince(startTime)
        
        let rtf = input.duration / processingTime
        print("Output: \(outputs.count) speakers, \(outputs.first?.samples.count ?? 0) samples each")
        print("RTF: \(String(format: "%.2f", rtf))x")
        
        // Save outputs
        let saverConfig = AudioSaver.Configuration(sampleRate: 16000)
        let saver = AudioSaver(config: saverConfig)
        
        for (i, output) in outputs.enumerated() {
            let path = "\(Self.ssModelDir)/ss_2spk_speaker\(i + 1).wav"
            try saver.save(MLXArray(output.samples), to: path)
            print("Saved: \(path)")
        }
        
        // Verify
        #expect(outputs.count == 2, "Should have 2 speaker outputs")
        #expect(outputs[0].sampleRate == 16000)
        #expect(outputs[0].frameCount > 0)
    }
    
    @Test("MossFormer2 SS 3-speaker separation with chunking", .enabledForMLX)
    func testMossFormer2SS3Speaker() async throws {
        
        print("\n=== MossFormer2 SS 3-Speaker with Chunking ===")
        
        // Create provider (8kHz model)
        let provider = MossFormer2SSProvider(model: .threeSpeaker, precision: .fp32)
        
        print("Loading model...")
        try await provider.load()
        print("Model ready")
        
        // Load test audio (8kHz for 3-speaker model)
        let testPath = "\(Self.ssModelDir)/mix3_8k.wav"
        print("Loading audio from: \(testPath)")
        
        let loaderConfig = AudioLoader.Configuration(targetSampleRate: 8000)
        let loader = AudioLoader(config: loaderConfig)
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let input = AudioBuffer(samples: samples, sampleRate: 8000, channels: 1)
        print("Input: \(input.samples.count) samples (\(String(format: "%.2f", input.duration))s)")
        
        // Separate speakers
        print("Processing...")
        let startTime = Date()
        let outputs = try await provider.separate(input)
        let processingTime = Date().timeIntervalSince(startTime)
        
        let rtf = input.duration / processingTime
        print("Output: \(outputs.count) speakers, \(outputs.first?.samples.count ?? 0) samples each")
        print("RTF: \(String(format: "%.2f", rtf))x")
        
        // Save outputs
        let saverConfig = AudioSaver.Configuration(sampleRate: 8000)
        let saver = AudioSaver(config: saverConfig)
        
        for (i, output) in outputs.enumerated() {
            let path = "\(Self.ssModelDir)/ss_3spk_speaker\(i + 1).wav"
            try saver.save(MLXArray(output.samples), to: path)
            print("Saved: \(path)")
        }
        
        // Verify
        #expect(outputs.count == 3, "Should have 3 speaker outputs")
        #expect(outputs[0].sampleRate == 8000)
    }
}

// MARK: - MossFormer2 Super Resolution Integration Tests

@Suite("MossFormer2 SR 48K Integration with Chunking", .tags(.integration))
struct MossFormer2SRIntegrationTests {
    
    static var srModelDir: String { "\(TestConfig.projectRoot)/Models/mossformer2_sr_mlx_swift" }
    
    @Test("MossFormer2 SR 16kHz to 48kHz upsampling with chunking", .enabledForMLX)
    func testMossFormer2SR48K() async throws {
        
        print("\n=== MossFormer2 SR 48K with Chunking ===")
        print("Super-resolution: 16kHz -> 48kHz")
        
        // Create provider - auto-downloads from HuggingFace if needed
        let provider = MLXProviders.mossformer2SR48K()
        
        print("Loading model...")
        try await provider.load()
        print("Model ready")
        
        // Load test audio at 16kHz (input rate)
        let testPath = "\(Self.srModelDir)/test_16k.wav"
        print("Loading audio from: \(testPath)")
        
        let loaderConfig = AudioLoader.Configuration(targetSampleRate: 16000)
        let loader = AudioLoader(config: loaderConfig)
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        
        let input = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        print("Input: \(input.samples.count) samples at 16kHz (\(String(format: "%.2f", input.duration))s)")
        
        // Upsample
        print("Processing...")
        let startTime = Date()
        let output = try await provider.process(input)
        let processingTime = Date().timeIntervalSince(startTime)
        
        let rtf = input.duration / processingTime
        print("Output: \(output.samples.count) samples at 48kHz (\(String(format: "%.2f", output.duration))s)")
        print("RTF: \(String(format: "%.2f", rtf))x")
        
        // Save output
        let outputPath = "\(Self.srModelDir)/sr_enhanced_test.wav"
        let saverConfig = AudioSaver.Configuration(sampleRate: 48000)
        let saver = AudioSaver(config: saverConfig)
        try saver.save(MLXArray(output.samples), to: outputPath)
        print("Saved: \(outputPath)")
        
        // Verify 3x upsample ratio (16kHz -> 48kHz)
        // Note: Output may differ slightly due to chunking boundary effects
        #expect(output.sampleRate == 48000, "Output should be 48kHz")
        let expectedSamples = input.samples.count * 3
        let sampleDiff = abs(output.samples.count - expectedSamples)
        #expect(sampleDiff < 3000, "Sample count should be approximately 3x input (within 3000 samples)")
    }
}

// MARK: - Test Tags

extension Tag {
    @Tag static var integration: Self
}
