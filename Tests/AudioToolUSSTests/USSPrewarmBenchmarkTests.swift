//
//  USSPrewarmBenchmarkTests.swift
//  AudioToolUSSTests
//
//  Benchmark tests to measure prewarming impact on embedding swap performance.
//  Tests whether prewarming with multiple embeddings improves chunked processing.
//

import XCTest
import AudioToolUSS
import AudioToolCore
import AudioUtils
import MLX
@preconcurrency import USSMLXSwift

final class USSPrewarmBenchmarkTests: XCTestCase {
    
    // Compute project root from source file path
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    // MARK: - Test Fixtures
    
    /// Load harry_potter_short.wav at 32kHz (USS native rate)
    private func loadHarryPotterAudio() throws -> AudioBuffer {
        let testPath = "\(Self.projectRoot)/Models/uss_mlx_swift/USSSwift/Samples/harry_potter_short.wav"
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 32000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        let samples = audio.asArray(Float.self)
        return AudioBuffer(samples: samples, sampleRate: 32000)
    }
    
    /// Split audio into chunks
    private func splitIntoChunks(_ audio: AudioBuffer, chunkSeconds: Double = 1.0) -> [AudioBuffer] {
        let chunkSize = Int(chunkSeconds * Double(audio.sampleRate))
        var chunks: [AudioBuffer] = []
        var offset = 0
        while offset < audio.samples.count {
            let end = min(offset + chunkSize, audio.samples.count)
            let chunkSamples = Array(audio.samples[offset..<end])
            chunks.append(AudioBuffer(samples: chunkSamples, sampleRate: audio.sampleRate))
            offset = end
        }
        return chunks
    }
    
    // MARK: - Prewarm Benchmark Tests
    
    /// Benchmark: Standard usage - prewarm with initial embedding only
    /// This is the current default behavior
    func testPrewarmSingleEmbedding_ChunkedProcessing() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("BENCHMARK: Single Embedding Prewarm (Default Behavior)")
        print(String(repeating: "=", count: 60))
        
        let audio = try loadHarryPotterAudio()
        let chunks = splitIntoChunks(audio, chunkSeconds: 1.0)
        print("Audio: \(String(format: "%.1f", Double(audio.samples.count) / 32000))s split into \(chunks.count) chunks")
        
        // Load USS with default behavior (prewarm with initial embedding only)
        let uss = USSProviders.speechSeparation(embeddingType: .speech)
        
        let loadStart = Date()
        try await uss.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Model load + prewarm: \(String(format: "%.3f", loadTime))s")
        
        // Process all chunks with alternating embeddings: music -> animal -> music -> animal...
        var timings: [(chunk: Int, embedding: String, time: Double)] = []
        
        let processStart = Date()
        for (idx, chunk) in chunks.enumerated() {
            // Alternate between music and animal
            let type: EmbeddingLoader.EmbeddingType = (idx % 2 == 0) ? .music : .animal
            
            let chunkStart = Date()
            _ = try await uss.process(chunk, type: type)
            let chunkTime = Date().timeIntervalSince(chunkStart)
            
            timings.append((chunk: idx, embedding: type.rawValue, time: chunkTime))
        }
        let totalTime = Date().timeIntervalSince(processStart)
        
        // Print per-chunk timings
        print("\nPer-chunk processing times:")
        for t in timings {
            print("  Chunk \(t.chunk + 1) [\(t.embedding)]: \(String(format: "%.4f", t.time))s")
        }
        
        // Calculate statistics
        let musicTimings = timings.filter { $0.embedding == "music" }.map { $0.time }
        let animalTimings = timings.filter { $0.embedding == "animal" }.map { $0.time }
        
        let avgMusic = musicTimings.reduce(0, +) / Double(musicTimings.count)
        let avgAnimal = animalTimings.reduce(0, +) / Double(animalTimings.count)
        
        // Check first vs subsequent calls for each type
        let firstMusic = musicTimings.first ?? 0
        let restMusic = musicTimings.dropFirst()
        let avgRestMusic = restMusic.isEmpty ? 0 : restMusic.reduce(0, +) / Double(restMusic.count)
        
        let firstAnimal = animalTimings.first ?? 0
        let restAnimal = animalTimings.dropFirst()
        let avgRestAnimal = restAnimal.isEmpty ? 0 : restAnimal.reduce(0, +) / Double(restAnimal.count)
        
        print("\nStatistics:")
        print("  Music  - First: \(String(format: "%.4f", firstMusic))s, Rest avg: \(String(format: "%.4f", avgRestMusic))s, Overall avg: \(String(format: "%.4f", avgMusic))s")
        print("  Animal - First: \(String(format: "%.4f", firstAnimal))s, Rest avg: \(String(format: "%.4f", avgRestAnimal))s, Overall avg: \(String(format: "%.4f", avgAnimal))s")
        print("  Total processing time: \(String(format: "%.3f", totalTime))s")
        
        // Calculate RTF
        let audioDuration = Double(audio.samples.count) / 32000.0
        let rtf = audioDuration / totalTime
        print("  RTF (2 embeddings per chunk): \(String(format: "%.1f", rtf))x")
        
        // Store results for comparison
        print("\n[RESULT] Single prewarm total: \(String(format: "%.4f", totalTime))s")
        
        await uss.unload()
    }
    
    /// Benchmark: Pre-prewarm with all target embeddings before processing
    /// Hypothesis: This should make first-call for each embedding faster
    func testPrewarmMultipleEmbeddings_ChunkedProcessing() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("BENCHMARK: Multiple Embedding Prewarm (Experimental)")
        print(String(repeating: "=", count: 60))
        
        let audio = try loadHarryPotterAudio()
        let chunks = splitIntoChunks(audio, chunkSeconds: 1.0)
        print("Audio: \(String(format: "%.1f", Double(audio.samples.count) / 32000))s split into \(chunks.count) chunks")
        
        // Load USS with speech, then manually prewarm with music and animal
        let uss = USSProviders.speechSeparation(embeddingType: .speech)
        
        let loadStart = Date()
        try await uss.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Model load + initial prewarm: \(String(format: "%.3f", loadTime))s")
        
        // Pre-prewarm: Run one dummy inference with each target embedding
        print("\nPre-prewarming with target embeddings...")
        let prewarmStart = Date()
        
        // Create a tiny chunk for prewarming (0.1 second)
        let tinyChunk = AudioBuffer(samples: [Float](repeating: 0, count: 3200), sampleRate: 32000)
        _ = try await uss.process(tinyChunk, type: .music)
        _ = try await uss.process(tinyChunk, type: .animal)
        
        let prewarmTime = Date().timeIntervalSince(prewarmStart)
        print("Additional prewarm time: \(String(format: "%.3f", prewarmTime))s")
        
        // Now process all chunks with alternating embeddings
        var timings: [(chunk: Int, embedding: String, time: Double)] = []
        
        let processStart = Date()
        for (idx, chunk) in chunks.enumerated() {
            // Alternate between music and animal
            let type: EmbeddingLoader.EmbeddingType = (idx % 2 == 0) ? .music : .animal
            
            let chunkStart = Date()
            _ = try await uss.process(chunk, type: type)
            let chunkTime = Date().timeIntervalSince(chunkStart)
            
            timings.append((chunk: idx, embedding: type.rawValue, time: chunkTime))
        }
        let totalTime = Date().timeIntervalSince(processStart)
        
        // Print per-chunk timings
        print("\nPer-chunk processing times:")
        for t in timings {
            print("  Chunk \(t.chunk + 1) [\(t.embedding)]: \(String(format: "%.4f", t.time))s")
        }
        
        // Calculate statistics
        let musicTimings = timings.filter { $0.embedding == "music" }.map { $0.time }
        let animalTimings = timings.filter { $0.embedding == "animal" }.map { $0.time }
        
        let avgMusic = musicTimings.reduce(0, +) / Double(musicTimings.count)
        let avgAnimal = animalTimings.reduce(0, +) / Double(animalTimings.count)
        
        // Check first vs subsequent calls for each type
        let firstMusic = musicTimings.first ?? 0
        let restMusic = musicTimings.dropFirst()
        let avgRestMusic = restMusic.isEmpty ? 0 : restMusic.reduce(0, +) / Double(restMusic.count)
        
        let firstAnimal = animalTimings.first ?? 0
        let restAnimal = animalTimings.dropFirst()
        let avgRestAnimal = restAnimal.isEmpty ? 0 : restAnimal.reduce(0, +) / Double(restAnimal.count)
        
        print("\nStatistics:")
        print("  Music  - First: \(String(format: "%.4f", firstMusic))s, Rest avg: \(String(format: "%.4f", avgRestMusic))s, Overall avg: \(String(format: "%.4f", avgMusic))s")
        print("  Animal - First: \(String(format: "%.4f", firstAnimal))s, Rest avg: \(String(format: "%.4f", avgRestAnimal))s, Overall avg: \(String(format: "%.4f", avgAnimal))s")
        print("  Total processing time: \(String(format: "%.3f", totalTime))s")
        print("  Total including prewarm: \(String(format: "%.3f", totalTime + prewarmTime))s")
        
        // Calculate RTF
        let audioDuration = Double(audio.samples.count) / 32000.0
        let rtf = audioDuration / totalTime
        print("  RTF (2 embeddings per chunk): \(String(format: "%.1f", rtf))x")
        
        print("\n[RESULT] Multi-prewarm total: \(String(format: "%.4f", totalTime))s (+ \(String(format: "%.4f", prewarmTime))s prewarm)")
        
        await uss.unload()
    }
    
    /// Benchmark: Use processMultiple for each chunk (batch both embeddings)
    /// This is the most efficient approach for your use case
    func testProcessMultiple_ChunkedProcessing() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("BENCHMARK: processMultiple (Recommended)")
        print(String(repeating: "=", count: 60))
        
        let audio = try loadHarryPotterAudio()
        let chunks = splitIntoChunks(audio, chunkSeconds: 1.0)
        print("Audio: \(String(format: "%.1f", Double(audio.samples.count) / 32000))s split into \(chunks.count) chunks")
        
        let uss = USSProviders.speechSeparation()
        
        let loadStart = Date()
        try await uss.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Model load + prewarm: \(String(format: "%.3f", loadTime))s")
        
        // Process all chunks with processMultiple
        var chunkTimings: [(chunk: Int, time: Double)] = []
        
        let processStart = Date()
        for (idx, chunk) in chunks.enumerated() {
            let chunkStart = Date()
            let results = try await uss.processMultiple(chunk, types: [.music, .animal])
            let chunkTime = Date().timeIntervalSince(chunkStart)
            
            // Verify we got both results
            XCTAssertNotNil(results[.music])
            XCTAssertNotNil(results[.animal])
            
            chunkTimings.append((chunk: idx, time: chunkTime))
        }
        let totalTime = Date().timeIntervalSince(processStart)
        
        // Print per-chunk timings
        print("\nPer-chunk processing times (both embeddings):")
        for t in chunkTimings {
            print("  Chunk \(t.chunk + 1): \(String(format: "%.4f", t.time))s")
        }
        
        // Statistics
        let avgChunkTime = chunkTimings.map { $0.time }.reduce(0, +) / Double(chunkTimings.count)
        let firstChunk = chunkTimings.first?.time ?? 0
        let restChunks = chunkTimings.dropFirst().map { $0.time }
        let avgRest = restChunks.isEmpty ? 0 : restChunks.reduce(0, +) / Double(restChunks.count)
        
        print("\nStatistics:")
        print("  First chunk: \(String(format: "%.4f", firstChunk))s")
        print("  Rest avg: \(String(format: "%.4f", avgRest))s")
        print("  Overall avg: \(String(format: "%.4f", avgChunkTime))s")
        print("  Total processing time: \(String(format: "%.3f", totalTime))s")
        
        // Calculate RTF
        let audioDuration = Double(audio.samples.count) / 32000.0
        let rtf = audioDuration / totalTime
        print("  RTF (2 embeddings per chunk): \(String(format: "%.1f", rtf))x")
        
        print("\n[RESULT] processMultiple total: \(String(format: "%.4f", totalTime))s")
        
        await uss.unload()
    }
    
    /// Run all three benchmarks and compare results
    func testCompareAllApproaches() async throws {
        print("\n" + String(repeating: "=", count: 70))
        print("USS EMBEDDING SWAP BENCHMARK COMPARISON")
        print("Testing: harry_potter_short.wav with [.music, .animal] per chunk")
        print(String(repeating: "=", count: 70))
        
        // Run single prewarm benchmark
        try await testPrewarmSingleEmbedding_ChunkedProcessing()
        
        // Clear GPU between tests
        GPU.clearCache()
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5s
        
        // Run multi-prewarm benchmark
        try await testPrewarmMultipleEmbeddings_ChunkedProcessing()
        
        // Clear GPU between tests
        GPU.clearCache()
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5s
        
        // Run processMultiple benchmark
        try await testProcessMultiple_ChunkedProcessing()
        
        print("\n" + String(repeating: "=", count: 70))
        print("COMPARISON COMPLETE")
        print("Check [RESULT] lines above to compare total times")
        print(String(repeating: "=", count: 70))
    }
    
    /// Detailed test: First-call penalty for each new embedding type
    func testFirstCallPenaltyPerEmbedding() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("BENCHMARK: First-Call Penalty Per Embedding Type")
        print(String(repeating: "=", count: 60))
        
        let audio = try loadHarryPotterAudio()
        let chunk = AudioBuffer(samples: Array(audio.samples.prefix(32000)), sampleRate: 32000) // 1 second
        print("Using 1s chunk for testing")
        
        let uss = USSProviders.speechSeparation(embeddingType: .speech)
        try await uss.load()
        
        // Test all 7 embedding types - measure first call for each
        let allTypes: [EmbeddingLoader.EmbeddingType] = [.speech, .music, .noise, .nature, .human, .animal, .things]
        
        print("\nFirst call timing for each embedding type:")
        for type in allTypes {
            let start = Date()
            _ = try await uss.process(chunk, type: type)
            let elapsed = Date().timeIntervalSince(start)
            print("  \(type.rawValue.padding(toLength: 8, withPad: " ", startingAt: 0)): \(String(format: "%.4f", elapsed))s")
        }
        
        // Now run second call for each - should be faster due to caching
        print("\nSecond call timing for each embedding type:")
        for type in allTypes {
            let start = Date()
            _ = try await uss.process(chunk, type: type)
            let elapsed = Date().timeIntervalSince(start)
            print("  \(type.rawValue.padding(toLength: 8, withPad: " ", startingAt: 0)): \(String(format: "%.4f", elapsed))s")
        }
        
        await uss.unload()
    }
    
    /// Test prewarmEmbeddings API
    func testPrewarmEmbeddingsAPI() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("TEST: prewarmEmbeddings API")
        print(String(repeating: "=", count: 60))
        
        let uss = USSProviders.speechSeparation(embeddingType: .speech)
        try await uss.load()
        
        // Prewarm with music and animal
        let prewarmStart = Date()
        try await uss.prewarmEmbeddings([.music, .animal])
        let prewarmTime = Date().timeIntervalSince(prewarmStart)
        print("Prewarm time for [.music, .animal]: \(String(format: "%.4f", prewarmTime))s")
        
        // Now test first call performance
        let audio = try loadHarryPotterAudio()
        let chunk = AudioBuffer(samples: Array(audio.samples.prefix(32000)), sampleRate: 32000)
        
        let musicStart = Date()
        _ = try await uss.process(chunk, type: .music)
        let musicTime = Date().timeIntervalSince(musicStart)
        
        let animalStart = Date()
        _ = try await uss.process(chunk, type: .animal)
        let animalTime = Date().timeIntervalSince(animalStart)
        
        print("After prewarm - Music: \(String(format: "%.4f", musicTime))s, Animal: \(String(format: "%.4f", animalTime))s")
        
        await uss.unload()
        
        // Just verify it ran without error
        XCTAssertLessThan(prewarmTime, 1.0, "Prewarm should complete in reasonable time")
    }
}
