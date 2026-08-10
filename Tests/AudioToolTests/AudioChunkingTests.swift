//
//  AudioChunkingTests.swift
//  AudioToolTests
//
//  Unit tests for AudioChunking
//

import Testing
import Foundation
@testable import AudioToolCore

@Suite("Audio Chunking")
struct AudioChunkingTests {
    
    // MARK: - Split Tests
    
    @Test("Split creates correct number of chunks")
    func testSplitChunkCount() {
        let config = ChunkingConfig(chunkDuration: 1.0, overlapRatio: 0.25, blendingStrategy: .triangular, sampleRate: 16000)
        let chunker = AudioChunker(config: config)
        
        // 4s audio, 1s chunks, 0.25 overlap = stride of 0.75s = ~6 chunks
        let samples = [Float](repeating: 1.0, count: 64000) // 4s
        let chunks = chunker.split(samples)
        
        #expect(chunks.count >= 5)
        #expect(chunks.first?.isFirst == true)
        #expect(chunks.last?.isLast == true)
    }
    
    @Test("Split handles audio shorter than chunk size")
    func testSplitShortAudio() {
        let config = ChunkingConfig(chunkDuration: 2.0, overlapRatio: 0.25, blendingStrategy: .triangular, sampleRate: 16000)
        let chunker = AudioChunker(config: config)
        
        let samples = [Float](repeating: 1.0, count: 16000) // 1s < 2s chunk
        let chunks = chunker.split(samples)
        
        #expect(chunks.count == 1)
        #expect(chunks[0].isFirst == true)
        #expect(chunks[0].isLast == true)
    }
    
    // MARK: - Reassemble Tests
    
    @Test("No overlap reassembly preserves length")
    func testNoOverlapReassembly() {
        let config = ChunkingConfig(chunkDuration: 1.0, overlapRatio: 0.0, blendingStrategy: .none, sampleRate: 16000)
        let chunker = AudioChunker(config: config)
        
        let samples = [Float](repeating: 1.0, count: 48000) // 3s
        let chunks = chunker.split(samples)
        let reassembled = chunker.reassemble(chunks, originalLength: samples.count)
        
        #expect(reassembled.count == samples.count)
    }
    
    @Test("Discard edges reassembly preserves length")
    func testDiscardEdgesReassembly() {
        let config = ChunkingConfig(chunkDuration: 1.0, overlapRatio: 0.25, blendingStrategy: .discardEdges, sampleRate: 16000)
        let chunker = AudioChunker(config: config)
        
        let samples = [Float](repeating: 1.0, count: 48000) // 3s
        let chunks = chunker.split(samples)
        let reassembled = chunker.reassemble(chunks, originalLength: samples.count)
        
        #expect(reassembled.count == samples.count)
    }
    
    @Test("Triangular reassembly preserves length")
    func testTriangularReassembly() {
        let config = ChunkingConfig(chunkDuration: 1.0, overlapRatio: 0.25, blendingStrategy: .triangular, sampleRate: 16000)
        let chunker = AudioChunker(config: config)
        
        let samples = [Float](repeating: 1.0, count: 48000) // 3s
        let chunks = chunker.split(samples)
        let reassembled = chunker.reassemble(chunks, originalLength: samples.count)
        
        #expect(reassembled.count == samples.count)
    }
    
    @Test("Hann window reassembly preserves length")
    func testHannReassembly() {
        let config = ChunkingConfig(chunkDuration: 1.0, overlapRatio: 0.5, blendingStrategy: .hann, sampleRate: 16000)
        let chunker = AudioChunker(config: config)
        
        let samples = [Float](repeating: 1.0, count: 48000) // 3s
        let chunks = chunker.split(samples)
        let reassembled = chunker.reassemble(chunks, originalLength: samples.count)
        
        #expect(reassembled.count == samples.count)
    }
    
    // MARK: - Roundtrip Tests
    
    @Test("Identity roundtrip with no processing")
    func testIdentityRoundtrip() {
        let config = ChunkingConfig(chunkDuration: 1.0, overlapRatio: 0.0, blendingStrategy: .none, sampleRate: 16000)
        let chunker = AudioChunker(config: config)
        
        // Create sine wave
        let samples: [Float] = (0..<32000).map { i in
            sin(Float(i) * 0.1)
        }
        
        let chunks = chunker.split(samples)
        let reassembled = chunker.reassemble(chunks, originalLength: samples.count)
        
        // Should be identical for no overlap
        var maxError: Float = 0
        for i in 0..<samples.count {
            maxError = max(maxError, abs(samples[i] - reassembled[i]))
        }
        
        #expect(maxError < 1e-6, "Perfect reconstruction expected for no overlap")
    }

    @Test("Every blending strategy identity-reassembles a constant tail")
    func testEveryStrategyIdentityRoundtrip() {
        let strategies: [ChunkBlendingStrategy] = [.none, .hann, .triangular, .discardEdges]
        let samples = [Float](repeating: 0.5, count: 2_350)

        for strategy in strategies {
            let overlap: Float = strategy == .none ? 0 : 0.5
            let config = ChunkingConfig(
                chunkDuration: 1,
                overlapRatio: overlap,
                blendingStrategy: strategy,
                sampleRate: 1_000
            )
            let chunker = AudioChunker(config: config)
            let chunks = chunker.split(samples)
            let output = chunker.reassemble(chunks, originalLength: samples.count)

            #expect(output == samples, "\(strategy) did not reconstruct its own chunks")
            #expect(chunks.last?.startIndex == 2_000,
                    "tail metadata should identify the first sample stored in the chunk")
        }
    }
    
    // MARK: - Preset Config Tests
    
    @Test("Preset configs have correct parameters")
    func testPresetConfigs() {
        let demucs = ChunkingConfig.demucs()
        #expect(demucs.chunkDuration == 7.8)
        #expect(demucs.overlapRatio == 0.25)
        
        let frcrn = ChunkingConfig.frcrnSE16K()
        #expect(frcrn.chunkDuration == 4.0)
        #expect(frcrn.overlapRatio == 0.25)
        
        let ganSE = ChunkingConfig.ganSECoreML()
        #expect(ganSE.chunkDuration == 1.594)
        #expect(ganSE.overlapRatio == 0.0)
        
        // 25%, not 50%: `generate.py` advances by `int(window * 0.75)`. The 0.75
        // is the stride, so the overlap is the remaining quarter. This asserted
        // 0.5 while the reference said 0.25, which is how the divergence survived.
        let sr48k = ChunkingConfig.mossformer2SR48K()
        #expect(sr48k.chunkDuration == 4.0)
        #expect(sr48k.overlapRatio == 0.25)
        #expect(sr48k.blendingStrategy == .discardEdges)
    }
}
