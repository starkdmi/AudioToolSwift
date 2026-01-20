//
//  KokoroVoiceMatcherTests.swift
//  ClearVoiceTests
//
//  Unit and integration tests for KokoroVoiceMatcher
//

import XCTest
@preconcurrency @testable import ClearVoiceTTS
@testable import ClearVoiceCore

final class KokoroVoiceMatcherTests: XCTestCase {
    
    // MARK: - VoiceEmbeddingTable Tests
    
    func testVoiceEmbeddingTableCreation() {
        let table = VoiceEmbeddingTable(
            voiceIds: ["voice1", "voice2"],
            embeddings: [
                [Float](repeating: 0.5, count: 256),
                [Float](repeating: -0.5, count: 256)
            ],
            calibrationText: "Test text"
        )
        
        XCTAssertEqual(table.count, 2)
        XCTAssertEqual(table.embeddingDimension, 256)
        XCTAssertTrue(table.isValid)
    }
    
    func testVoiceEmbeddingTableLookup() {
        let embedding1 = [Float](repeating: 0.1, count: 256)
        let embedding2 = [Float](repeating: 0.2, count: 256)
        
        let table = VoiceEmbeddingTable(
            voiceIds: ["af_bella", "am_echo"],
            embeddings: [embedding1, embedding2],
            calibrationText: "Test"
        )
        
        let found = table.embedding(for: "af_bella")
        XCTAssertNotNil(found)
        XCTAssertEqual(found?[0], 0.1)
        
        let notFound = table.embedding(for: "unknown_voice")
        XCTAssertNil(notFound)
    }
    
    func testVoiceEmbeddingTableSerialization() throws {
        let table = VoiceEmbeddingTable(
            voiceIds: ["voice1", "voice2"],
            embeddings: [
                [Float](repeating: 0.5, count: 256),
                [Float](repeating: -0.5, count: 256)
            ],
            calibrationText: "Test text"
        )
        
        // Create temp URL
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_embeddings.json")
        
        // Save
        try table.save(to: tempURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
        
        // Load
        let loaded = try VoiceEmbeddingTable.load(from: tempURL)
        XCTAssertEqual(loaded.voiceIds, table.voiceIds)
        XCTAssertEqual(loaded.count, table.count)
        XCTAssertEqual(loaded.calibrationText, table.calibrationText)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testVoiceEmbeddingTableValidation() {
        // Empty table is invalid
        let emptyTable = VoiceEmbeddingTable(
            voiceIds: [],
            embeddings: [],
            calibrationText: ""
        )
        XCTAssertFalse(emptyTable.isValid)
        
        // Mismatched counts is invalid
        let mismatchedTable = VoiceEmbeddingTable(
            voiceIds: ["v1", "v2"],
            embeddings: [[Float](repeating: 0, count: 256)],  // Only 1 embedding
            calibrationText: ""
        )
        XCTAssertFalse(mismatchedTable.isValid)
    }
    
    // MARK: - VoiceMatchResult Tests
    
    func testVoiceMatchResultPrimaryVoice() {
        let result = VoiceMatchResult(
            weights: [("af_bella", 0.7), ("am_echo", 0.3)],
            similarity: 0.95,
            matchTime: 0.1
        )
        
        XCTAssertEqual(result.primaryVoice, "af_bella")
        XCTAssertEqual(result.weightsDictionary["af_bella"], 0.7)
        XCTAssertEqual(result.weightsDictionary["am_echo"], 0.3)
    }
    
    // MARK: - Voice Matching Algorithm Tests
    
    func testMatchVoiceWithMockedEmbeddings() async throws {
        let matcher = KokoroVoiceMatcher(topK: 3)
        
        // Create mock voice embeddings
        // Voice 1: mostly positive values
        var v1 = [Float](repeating: 0, count: 256)
        for i in 0..<256 { v1[i] = Float(i) / 256.0 }
        
        // Voice 2: mostly negative values  
        var v2 = [Float](repeating: 0, count: 256)
        for i in 0..<256 { v2[i] = -Float(i) / 256.0 }
        
        // Voice 3: mixed values
        var v3 = [Float](repeating: 0, count: 256)
        for i in 0..<256 { v3[i] = sin(Float(i) * 0.1) }
        
        let table = VoiceEmbeddingTable(
            voiceIds: ["voice1", "voice2", "voice3"],
            embeddings: [v1, v2, v3],
            calibrationText: "Test"
        )
        
        // Reference is identical to voice1
        let referenceEmbedding = v1
        
        // Mock embedding extractor that returns the preconfigured reference
        let result = try await matcher.matchVoice(
            referenceAudio: [Float](repeating: 0, count: 16000),  // Dummy audio
            embeddingTable: table,
            extractEmbedding: { _ in
                // Return the reference embedding regardless of input
                return referenceEmbedding
            }
        )
        
        // Should strongly prefer voice1 since reference == voice1
        XCTAssertEqual(result.primaryVoice, "voice1")
        XCTAssertGreaterThan(result.similarity, 0.9)
        
        // Weights should be dominated by voice1
        let voice1Weight = result.weights.first { $0.name == "voice1" }?.weight ?? 0
        XCTAssertGreaterThan(voice1Weight, 0.5)
    }
    
    func testMatchVoiceWithBlendedReference() async throws {
        let matcher = KokoroVoiceMatcher(topK: 3)
        
        // Orthogonal basis vectors (easy to solve)
        var v1 = [Float](repeating: 0, count: 256)
        v1[0] = 1.0  // Unit vector on axis 0
        
        var v2 = [Float](repeating: 0, count: 256)
        v2[1] = 1.0  // Unit vector on axis 1
        
        var v3 = [Float](repeating: 0, count: 256)
        v3[2] = 1.0  // Unit vector on axis 2
        
        let table = VoiceEmbeddingTable(
            voiceIds: ["voice1", "voice2", "voice3"],
            embeddings: [v1, v2, v3],
            calibrationText: "Test"
        )
        
        // Reference is 70% voice1 + 30% voice2
        var refEmbedding = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            refEmbedding[i] = 0.7 * v1[i] + 0.3 * v2[i]
        }
        let referenceEmbedding = refEmbedding
        
        let result = try await matcher.matchVoice(
            referenceAudio: [Float](repeating: 0, count: 16000),
            embeddingTable: table,
            extractEmbedding: { _ in referenceEmbedding }
        )
        
        // Should find weights close to [0.7, 0.3, 0.0]
        let weightsDict = result.weightsDictionary
        
        let w1 = weightsDict["voice1"] ?? 0
        let w2 = weightsDict["voice2"] ?? 0
        
        XCTAssertEqual(w1, 0.7, accuracy: 0.15, "voice1 weight should be ~0.7")
        XCTAssertEqual(w2, 0.3, accuracy: 0.15, "voice2 weight should be ~0.3")
    }
    
    func testMatchVoicePerformance() async throws {
        let matcher = KokoroVoiceMatcher(topK: 5)
        
        // Create realistic 10-voice table
        var embeddings: [[Float]] = []
        for i in 0..<10 {
            var vec = [Float](repeating: 0, count: 256)
            for j in 0..<256 {
                vec[j] = Float.random(in: -1...1) + Float(i) * 0.1
            }
            embeddings.append(vec)
        }
        
        let table = VoiceEmbeddingTable(
            voiceIds: (0..<10).map { "voice\($0)" },
            embeddings: embeddings,
            calibrationText: "Test"
        )
        
        var refEmb = [Float](repeating: 0, count: 256)
        for j in 0..<256 {
            refEmb[j] = Float.random(in: -1...1)
        }
        let referenceEmbedding = refEmb
        
        // Measure performance
        let start = Date()
        
        for _ in 0..<100 {
            _ = try await matcher.matchVoice(
                referenceAudio: [],
                embeddingTable: table,
                extractEmbedding: { _ in referenceEmbedding }
            )
        }
        
        let elapsed = Date().timeIntervalSince(start)
        let avgTime = elapsed / 100.0
        
        // Use CI-aware threshold - CI machines may be slower
        let isCI = ProcessInfo.processInfo.environment["CI"] == "1"
        let maxTimeMs = isCI ? 20.0 : 5.0
        XCTAssertLessThan(avgTime, maxTimeMs / 1000.0, "Matching should take < \(maxTimeMs)ms (got \(avgTime * 1000)ms)")
    }
}
