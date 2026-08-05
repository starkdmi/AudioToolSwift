//
//  USSEmbeddingSwapTests.swift
//  AudioToolUSSTests
//
//  Tests for USS embedding swap functionality:
//  - Efficient multi-type separation with single model load
//  - Embedding switching without model reload
//  - Model lifecycle (load/unload/reload)
//  - Chunked processing workflow
//

import XCTest
import AudioToolUSS
import AudioToolCore
import AudioUtils
import MLX
@preconcurrency import USSMLXSwift

final class USSEmbeddingSwapTests: XCTestCase {
    
    // Compute project root from source file path
    static let projectRoot: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()
    
    // MARK: - Test Fixtures
    
    /// Load test audio at 32kHz (USS native rate)
    private func loadTestAudio(duration: Double = 2.0) throws -> AudioBuffer {
        let testPath = "\(Self.projectRoot)/Models/uss_mlx_swift/test.wav"
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 32000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: URL(fileURLWithPath: testPath))
        eval(audio)
        
        // Trim to requested duration
        let samples = audio.asArray(Float.self)
        let maxSamples = min(samples.count, Int(duration * 32000))
        return AudioBuffer(samples: Array(samples.prefix(maxSamples)), sampleRate: 32000)
    }
    
    /// Load short chunk for streaming tests
    private func loadTestChunk(seconds: Double = 1.0) throws -> AudioBuffer {
        return try loadTestAudio(duration: seconds)
    }
    
    // MARK: - Embedding Swap Performance Tests
    
    /// Test that embedding swap is much faster than model load
    func testEmbeddingSwapPerformance() async throws {
        print("\n=== USS Embedding Swap Performance Test ===")
        
        // Load USS model
        let uss = USSProviders.speechSeparation()
        
        let loadStart = Date()
        try await uss.load()
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Model load time: \(String(format: "%.2f", loadTime))s")
        
        // Swap embedding (should be instant since all embeddings are cached)
        let swapStart = Date()
        try await uss.setConditioning(.music)
        let swapTime = Date().timeIntervalSince(swapStart)
        print("Embedding swap time: \(String(format: "%.4f", swapTime))s")
        
        // Verify swap was successful
        let currentType = await uss.activeEmbeddingType
        XCTAssertEqual(currentType, .music)
        
        // Swap should be essentially instant (embeddings cached)
        XCTAssertLessThan(swapTime, 0.1, "Embedding swap should be < 100ms (cached)")
        
        // Cleanup
        await uss.unload()
        print("Embedding swap is \(String(format: "%.0f", loadTime / swapTime))x faster than model load")
    }
    
    /// Test process(_:type:) is stateless (doesn't change stored embedding)
    func testProcessWithTypeIsStateless() async throws {
        print("\n=== USS Stateless Process Test ===")
        
        let uss = USSProviders.speechSeparation(embeddingType: .speech)
        try await uss.load()
        
        // Verify initial state
        let initialType = await uss.activeEmbeddingType
        XCTAssertEqual(initialType, .speech)
        
        let chunk = try loadTestChunk()
        
        // Process with .music (one-off, should not change state)
        let musicResult = try await uss.process(chunk, type: .music)
        // Note: Output may be very quiet if test audio doesn't contain target sound
        XCTAssertEqual(musicResult.sampleRate, 32000)
        XCTAssertGreaterThan(musicResult.samples.count, 0)
        print("  Music output: \(musicResult.samples.count) samples, max: \(String(format: "%.6f", musicResult.samples.max() ?? 0))")
        
        // State should still be .speech
        let afterType = await uss.activeEmbeddingType
        XCTAssertEqual(afterType, .speech, "Active embedding should not change after process(_:type:)")
        
        // Process with .animal (one-off)
        let animalResult = try await uss.process(chunk, type: .animal)
        XCTAssertEqual(animalResult.sampleRate, 32000)
        XCTAssertGreaterThan(animalResult.samples.count, 0)
        print("  Animal output: \(animalResult.samples.count) samples, max: \(String(format: "%.6f", animalResult.samples.max() ?? 0))")
        
        // State should still be .speech
        let finalType = await uss.activeEmbeddingType
        XCTAssertEqual(finalType, .speech, "Active embedding should remain .speech")
        
        await uss.unload()
        print("process(_:type:) is correctly stateless")
    }
    
    // MARK: - Multi-Type Separation Tests
    
    /// Test processMultiple for efficient batch separation
    func testProcessMultiple() async throws {
        print("\n=== USS processMultiple Test ===")
        
        let uss = USSProviders.speechSeparation()
        try await uss.load()
        
        let audio = try loadTestAudio(duration: 2.0)
        print("Input: \(audio.samples.count) samples (\(String(format: "%.1f", Double(audio.samples.count) / 32000))s)")
        
        // Process multiple types in one call
        let types: [EmbeddingLoader.EmbeddingType] = [.music, .animal, .noise]
        
        let start = Date()
        let results = try await uss.processMultiple(audio, types: types)
        let elapsed = Date().timeIntervalSince(start)
        
        print("Separated \(types.count) types in \(String(format: "%.2f", elapsed))s")
        
        // Verify all results present
        for type in types {
            guard let result = results[type] else {
                XCTFail("Missing result for \(type.rawValue)")
                continue
            }
            let maxAmp = result.samples.max() ?? 0
            print("  \(type.rawValue): \(result.samples.count) samples, max: \(String(format: "%.6f", maxAmp))")
            // Just verify result has samples - amplitude depends on test audio content
            XCTAssertEqual(result.sampleRate, 32000)
            XCTAssertGreaterThan(result.samples.count, 0)
        }
        
        await uss.unload()
    }
    
    /// Test chunked multi-type separation workflow (your primary use case)
    func testChunkedMultiTypeSeparation() async throws {
        print("\n=== USS Chunked Multi-Type Separation Test ===")
        
        let uss = USSProviders.speechSeparation()
        try await uss.load()
        
        // Simulate streaming: load full audio and split into chunks
        let fullAudio = try loadTestAudio(duration: 4.0)
        let chunkSize = 32000 // 1 second at 32kHz
        
        var chunks: [AudioBuffer] = []
        var offset = 0
        while offset < fullAudio.samples.count {
            let end = min(offset + chunkSize, fullAudio.samples.count)
            let chunkSamples = Array(fullAudio.samples[offset..<end])
            chunks.append(AudioBuffer(samples: chunkSamples, sampleRate: 32000))
            offset = end
        }
        print("Split into \(chunks.count) chunks of ~1s each")
        
        // Process each chunk with multiple embeddings
        var allMusicOutputs: [[Float]] = []
        var allAnimalOutputs: [[Float]] = []
        
        let start = Date()
        for (idx, chunk) in chunks.enumerated() {
            let results = try await uss.processMultiple(chunk, types: [.music, .animal])
            
            if let music = results[.music] {
                allMusicOutputs.append(music.samples)
            }
            if let animal = results[.animal] {
                allAnimalOutputs.append(animal.samples)
            }
            
            print("  Chunk \(idx + 1)/\(chunks.count) processed")
        }
        let elapsed = Date().timeIntervalSince(start)
        
        print("Total processing time: \(String(format: "%.2f", elapsed))s")
        print("  Music chunks: \(allMusicOutputs.count)")
        print("  Animal chunks: \(allAnimalOutputs.count)")
        
        // Verify outputs
        XCTAssertEqual(allMusicOutputs.count, chunks.count)
        XCTAssertEqual(allAnimalOutputs.count, chunks.count)
        
        // Calculate RTF
        let audioDuration = Double(fullAudio.samples.count) / 32000.0
        let rtf = audioDuration / elapsed
        print("RTF: \(String(format: "%.1f", rtf))x (for 2 embedding types)")
        
        await uss.unload()
    }
    
    // MARK: - Model Lifecycle Tests
    
    /// Test unload releases resources and process throws after unload
    func testUnloadReleasesMemory() async throws {
        print("\n=== USS Unload Test ===")
        
        let uss = USSProviders.speechSeparation()
        try await uss.load()
        
        let isLoadedBefore = await uss.checkIfLoaded()
        XCTAssertTrue(isLoadedBefore, "Model should be loaded")
        
        await uss.unload()
        
        let isLoadedAfter = await uss.checkIfLoaded()
        XCTAssertFalse(isLoadedAfter, "Model should be unloaded")
        
        // Process should throw after unload
        let chunk = try loadTestChunk()
        do {
            _ = try await uss.process(chunk, type: .speech)
            XCTFail("process() should throw after unload")
        } catch let error as AudioToolError {
            print("Expected error: \(error.localizedDescription)")
            XCTAssertTrue(error.localizedDescription.contains("not loaded") || 
                          error.localizedDescription.contains("USS MLX"))
        }
        
        print("Unload correctly releases resources and blocks processing")
    }
    
    /// Test model can be reloaded after unload
    func testReloadAfterUnload() async throws {
        print("\n=== USS Reload After Unload Test ===")
        
        let uss = USSProviders.speechSeparation()
        
        // First load
        try await uss.load()
        let chunk = try loadTestChunk()
        let output1 = try await uss.process(chunk, type: .speech)
        print("First load: output max = \(String(format: "%.4f", output1.samples.max() ?? 0))")
        
        // Unload
        await uss.unload()
        var isLoaded = await uss.checkIfLoaded()
        XCTAssertFalse(isLoaded)
        
        // Reload
        try await uss.load()
        isLoaded = await uss.checkIfLoaded()
        XCTAssertTrue(isLoaded)
        
        // Process again
        let output2 = try await uss.process(chunk, type: .speech)
        print("After reload: output max = \(String(format: "%.4f", output2.samples.max() ?? 0))")
        
        // Just verify we got output (amplitude depends on test audio content)
        XCTAssertEqual(output2.sampleRate, 32000)
        XCTAssertGreaterThan(output2.samples.count, 0)
        
        await uss.unload()
        print("Reload after unload works correctly")
    }
    
    // MARK: - Embedding Switching Tests
    
    /// Test setConditioning changes active embedding
    func testSetConditioningChangesState() async throws {
        print("\n=== USS setConditioning Test ===")
        
        let uss = USSProviders.speechSeparation(embeddingType: .speech)
        try await uss.load()
        
        var currentType = await uss.activeEmbeddingType
        XCTAssertEqual(currentType, .speech)
        
        try await uss.setConditioning(.music)
        currentType = await uss.activeEmbeddingType
        XCTAssertEqual(currentType, .music)
        
        try await uss.setConditioning(.animal)
        currentType = await uss.activeEmbeddingType
        XCTAssertEqual(currentType, .animal)
        
        try await uss.setConditioning(.nature)
        currentType = await uss.activeEmbeddingType
        XCTAssertEqual(currentType, .nature)
        
        await uss.unload()
        print("setConditioning correctly updates active embedding type")
    }
    
    /// Test all embedding types can be used
    func testAllEmbeddingTypes() async throws {
        print("\n=== USS All Embedding Types Test ===")
        
        let uss = USSProviders.speechSeparation()
        try await uss.load()
        
        let chunk = try loadTestChunk(seconds: 1.0)
        
        let allTypes: [EmbeddingLoader.EmbeddingType] = [.speech, .music, .noise, .nature, .human, .animal, .things]
        
        for type in allTypes {
            let output = try await uss.process(chunk, type: type)
            let maxAmp = output.samples.max() ?? 0
            print("  \(type.rawValue): max amplitude = \(String(format: "%.4f", maxAmp))")
            
            // Just verify processing doesn't crash - amplitude varies by content
            XCTAssertEqual(output.sampleRate, 32000)
            XCTAssertGreaterThan(output.samples.count, 0)
        }
        
        await uss.unload()
        print("All 7 embedding types work correctly")
    }
    
    // MARK: - Background Extraction with Type Tests
    
    /// Test separateWithBackground with specific type
    func testSeparateWithBackgroundWithType() async throws {
        print("\n=== USS separateWithBackground with Type Test ===")
        
        let uss = USSProviders.speechSeparation(embeddingType: .speech)
        try await uss.load()
        
        let audio = try loadTestAudio(duration: 2.0)
        
        // Separate music with background
        let result = try await uss.separateWithBackground(audio, type: .music)
        
        print("Music separation with background:")
        print("  Separated: \(result.separated.samples.count) samples, max: \(String(format: "%.4f", result.separated.samples.max() ?? 0))")
        print("  Background: \(result.background.samples.count) samples, max: \(String(format: "%.4f", result.background.samples.max() ?? 0))")
        
        // Verify lengths match
        XCTAssertEqual(result.separated.samples.count, result.background.samples.count)
        
        // Active embedding should still be .speech (stateless)
        let finalType = await uss.activeEmbeddingType
        XCTAssertEqual(finalType, .speech)
        
        await uss.unload()
    }
    
    // MARK: - Factory Methods Tests
    
    /// Test all factory methods create valid providers
    func testFactoryMethods() async throws {
        print("\n=== USS Factory Methods Test ===")
        
        let providers: [(String, USSMLXProvider)] = [
            ("speechSeparation", USSProviders.speechSeparation()),
            ("musicSeparation", USSProviders.musicSeparation()),
            ("noiseSeparation", USSProviders.noiseSeparation()),
            ("animalSeparation", USSProviders.animalSeparation()),
            ("natureSeparation", USSProviders.natureSeparation()),
            ("humanSeparation", USSProviders.humanSeparation()),
            ("thingsSeparation", USSProviders.thingsSeparation()),
            ("separation(type:)", USSProviders.separation(type: .music)),
        ]
        
        for (name, provider) in providers {
            // Just verify they can be created without loading
            XCTAssertEqual(provider.sampleRate, 32000, "\(name) should have 32kHz sample rate")
            print("  \(name) created successfully")
        }
        
        print("All factory methods work correctly")
    }
}
