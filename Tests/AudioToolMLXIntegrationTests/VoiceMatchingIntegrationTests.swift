//
//  VoiceMatchingIntegrationTests.swift
//  AudioToolMLXIntegrationTests
//
//  Integration tests for voice matching with real audio files
//

import XCTest
import AudioToolTestSupport
import AudioTool
import AudioToolCore
@preconcurrency import AudioToolTTS
@preconcurrency import AudioToolFluidAudio
@preconcurrency import MLX
@preconcurrency import AudioUtils

final class VoiceMatchingIntegrationTests: IntegrationTestCase {
    
    // Compute project root from source file path
    
    // NOTE: This test requires Metal/MLX and must be run via xcodebuild
    // Run: xcodebuild test -scheme AudioToolSwift-Package -destination 'platform=macOS' -derivedDataPath .build/DerivedData -only-testing:AudioToolMLXIntegrationTests/VoiceMatchingIntegrationTests
    
    /// Test full voice matching pipeline with real audio
    /// 
    /// Reference files (relative to project root):
    /// - Docs/burunow_short.wav
    /// - Docs/reference.wav  
    /// - Docs/watson_short.wav
    func testVoiceMatchingWithRealAudio() async throws {
        print("\n=== Voice Matching Integration Test ===\n")
        
        // Reference audio files (at project root /Docs/, not /AudioTool/Docs/)
        let referenceFiles = try [
            "Docs/burunow_short.wav",
            "Docs/reference.wav",
            "Docs/watson_short.wav"
        ].map { try reference($0).path }
        
        // Output directory - a scratch path, not the process's working directory
        let outputDir = try outputDirectory().path
        
        // Load Kokoro TTS model – if the model cannot be downloaded we skip the test
        print("Loading Kokoro TTS model (optional)...")
        let downloader = ModelDownloader.shared
        var modelPath: URL? = nil
        do {
            let requiredFiles = [
                "*.safetensors",
                "voices/*.safetensors",
                "config.json",
            ]
            if let cached = downloader.localPath(
                for: "mlx-community/Kokoro-82M-bf16",
                matching: requiredFiles
            ) {
                modelPath = cached
            } else {
                modelPath = try await downloader.downloadAndGetPath(
                    repo: "mlx-community/Kokoro-82M-bf16",
                    matching: requiredFiles
                ) { progress in
                    print("Downloading: \(Int(progress.fractionCompleted * 100))%")
                }
            }
        } catch {
            // If we cannot obtain the model (e.g., no internet), skip the test gracefully
            throw XCTSkip("Kokoro model not available – skipping voice-matching test: \(error)")
        }
        guard let modelURL = modelPath else {
            throw XCTSkip("Kokoro model path is nil – skipping test")
        }
        let voicesDir = modelURL.appendingPathComponent("voices")
        let tts = KokoroTTSProvider(modelPath: modelURL, language: .americanEnglish)
        try await tts.load()
        try await tts.loadVoices(from: voicesDir)
        print("✓ Loaded \(tts.availableVoices.count) voices\n")
        
        // Load speaker embedding provider
        print("Loading speaker embedding model...")
        let embeddingProvider = SpeakerEmbeddingProvider()
        try await embeddingProvider.load()
        print("✓ Embedding model ready\n")
        
        // Create voice matcher
        let matcher = KokoroVoiceMatcher(topK: 5)
        
        // Precompute voice embeddings
        print("=== Precomputing Voice Embeddings ===")
        let precomputeStart = Date()
        let embeddingTable = try await matcher.precomputeEmbeddings(
            tts: tts,
            extractEmbedding: { audio in
                let audio16k = resampleTo16kHz(audio, from: 24000)
                return try await embeddingProvider.extractEmbedding(audio16k)
            }
        )
        let precomputeTime = Date().timeIntervalSince(precomputeStart)
        print("✓ Precomputed \(embeddingTable.count) embeddings in \(String(format: "%.1f", precomputeTime))s")
        
        // Save embedding table
        let tablePath = URL(fileURLWithPath: outputDir).appendingPathComponent("voice_embeddings.json")
        try embeddingTable.save(to: tablePath)
        print("✓ Saved: voice_embeddings.json\n")
        
        // Match each reference file
        print("=== Matching Reference Audio ===\n")
        
        let testText = "Hello! This is a test of voice matching. The system found the best blend of Kokoro voices to approximate my speaking style."
        
        for refPath in referenceFiles {
            let filename = URL(fileURLWithPath: refPath).deletingPathExtension().lastPathComponent
            print("--- \(filename) ---")
            
            // Extract reference embedding
            let matchStart = Date()
            let refEmbedding = try await embeddingProvider.extractEmbedding(from: URL(fileURLWithPath: refPath))
            
            // Match
            let result = try await matcher.matchVoice(
                referenceAudio: [],
                embeddingTable: embeddingTable,
                extractEmbedding: { _ in refEmbedding }
            )
            let matchTime = Date().timeIntervalSince(matchStart)
            
            // Validate results
            XCTAssertGreaterThan(result.weights.count, 0, "Should find at least one matching voice")
            XCTAssertGreaterThan(result.similarity, 0, "Similarity should be positive")
            
            let totalWeight = result.weights.map(\.weight).reduce(0, +)
            XCTAssertEqual(totalWeight, 1.0, accuracy: 0.01, "Weights should sum to 1.0")
            
            for (_, weight) in result.weights {
                XCTAssertGreaterThanOrEqual(weight, 0, "All weights should be non-negative")
            }
            
            // Print results
            print("Match time: \(String(format: "%.0f", matchTime * 1000))ms")
            print("Similarity: \(String(format: "%.1f%%", result.similarity * 100))")
            print("Voice blend:")
            for (voice, weight) in result.weights {
                let bar = String(repeating: "█", count: Int(weight * 20))
                print("  \(voice.padding(toLength: 12, withPad: " ", startingAt: 0)) \(String(format: "%5.1f%%", weight * 100)) \(bar)")
            }
            
            // Generate output audio
            print("\nSynthesizing...")
            let synthStart = Date()
            let blendedVoice = try await tts.blendedVoice(from: result)
            let audio = try await tts.synthesize(testText, voiceEmbedding: blendedVoice)
            let synthTime = Date().timeIntervalSince(synthStart)
            print("Synthesis: \(String(format: "%.2f", audio.duration))s audio in \(String(format: "%.2f", synthTime))s")
            
            // Save
            let outputPath = "\(outputDir)/matched_\(filename).wav"
            let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
            try saver.save(MLXArray(audio.samples), to: outputPath)
            print("✓ Saved: matched_\(filename).wav\n")
        }
        
        // List output files
        print("=== Voice Matching Complete ===")
        print("Output: \(outputDir)")
        if let files = try? FileManager.default.contentsOfDirectory(atPath: outputDir) {
            print("\nGenerated files:")
            for file in files.sorted() where file.hasSuffix(".wav") || file.hasSuffix(".json") {
                print("  - \(file)")
            }
        }
        
        GPU.clearCache()
    }
}

// Helper: Resample to 16kHz (global to avoid capturing non-Sendable self)
private func resampleTo16kHz(_ samples: [Float], from sourceRate: Int) -> [Float] {
    guard sourceRate != 16000 else { return samples }
    
    let ratio = Float(16000) / Float(sourceRate)
    let outputLength = Int(Float(samples.count) * ratio)
    guard outputLength > 0 else { return [] }
    
    var output = [Float](repeating: 0, count: outputLength)
    
    for i in 0..<outputLength {
        let srcPos = Float(i) / ratio
        let srcIdx = Int(srcPos)
        let frac = srcPos - Float(srcIdx)
        
        let idx0 = min(srcIdx, samples.count - 1)
        let idx1 = min(srcIdx + 1, samples.count - 1)
        
        output[i] = samples[idx0] * (1 - frac) + samples[idx1] * frac
    }
    
    return output
}
