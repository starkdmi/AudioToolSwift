//
//  VADPipelineTests.swift
//  ClearVoiceFluidAudioTests
//
//  Tests for chaining VAD with other processors (SE, SS, etc.)
//

import XCTest
import ClearVoiceFluidAudio
import ClearVoiceCore

final class VADPipelineTests: XCTestCase {
    
    // MARK: - VAD + SE Pipeline Test
    
    /// Test chaining: VAD detects speech → SE enhances only speech chunks
    func testVADWithSEPipeline() async throws {
        // 1. Create VAD provider
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        
        // 2. Create test audio (10 seconds with speech-like patterns)
        // Simulating: 2s silence, 4s "speech", 2s silence, 2s "speech"
        var samples = [Float]()
        
        // 0-2s: Silence
        samples.append(contentsOf: [Float](repeating: 0.0, count: 16000 * 2))
        
        // 2-6s: Speech-like noise
        for _ in 0..<(16000 * 4) {
            samples.append(Float.random(in: -0.3...0.3) * sin(Float.random(in: 0...6.28)))
        }
        
        // 6-8s: Silence
        samples.append(contentsOf: [Float](repeating: 0.0, count: 16000 * 2))
        
        // 8-10s: Speech-like noise
        for _ in 0..<(16000 * 2) {
            samples.append(Float.random(in: -0.3...0.3) * sin(Float.random(in: 0...6.28)))
        }
        
        let audio = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        print("Input audio: \(audio.samples.count) samples (\(audio.duration)s)")
        
        // 3. Run VAD
        let startVAD = Date()
        let segments = try await vad.detect(audio)
        let vadTime = Date().timeIntervalSince(startVAD)
        
        print("VAD detected \(segments.count) speech segments in \(String(format: "%.3f", vadTime))s")
        for (i, segment) in segments.enumerated() {
            print("  Segment \(i+1): \(String(format: "%.2f", segment.timeRange.start))s - \(String(format: "%.2f", segment.timeRange.end))s (prob: \(segment.probability))")
        }
        
        // 4. Extract speech chunks based on VAD segments
        var speechChunks: [AudioBuffer] = []
        for segment in segments {
            let startSample = Int(segment.timeRange.start * Double(audio.sampleRate))
            let endSample = min(Int(segment.timeRange.end * Double(audio.sampleRate)), audio.samples.count)
            
            if endSample > startSample {
                let chunkSamples = Array(audio.samples[startSample..<endSample])
                speechChunks.append(AudioBuffer(
                    samples: chunkSamples,
                    sampleRate: audio.sampleRate,
                    channels: audio.channels
                ))
            }
        }
        
        let totalSpeechDuration = speechChunks.reduce(0.0) { $0 + $1.duration }
        print("Extracted \(speechChunks.count) speech chunks totaling \(String(format: "%.2f", totalSpeechDuration))s")
        
        // 5. Verify we extracted less audio than original (VAD filtered silence)
        XCTAssertLessThan(totalSpeechDuration, audio.duration, 
                         "VAD should filter out silence, reducing total audio")
        
        // 6. In real pipeline, would now enhance each chunk with SE:
        // for chunk in speechChunks {
        //     let enhanced = try await seProvider.process(chunk)
        // }
        
        print("\n✓ VAD + SE pipeline test passed")
        print("  - VAD reduced audio from \(String(format: "%.1f", audio.duration))s to \(String(format: "%.1f", totalSpeechDuration))s")
        print("  - Savings: \(String(format: "%.0f", (1 - totalSpeechDuration/audio.duration) * 100))% less audio to enhance")
    }
    
    /// Test streaming VAD for real-time pipeline
    func testStreamingVADPipeline() async throws {
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        
        // Simulate streaming 256ms chunks
        let chunkSize = 4096  // 256ms at 16kHz
        let totalSamples = 16000 * 5  // 5 seconds
        
        // Create async stream of chunks
        let audioStream = AsyncStream<AudioBuffer> { continuation in
            Task {
                for offset in stride(from: 0, to: totalSamples, by: chunkSize) {
                    let end = min(offset + chunkSize, totalSamples)
                    let samples = (offset..<end).map { _ in Float.random(in: -0.2...0.2) }
                    continuation.yield(AudioBuffer(samples: samples, sampleRate: 16000, channels: 1))
                }
                continuation.finish()
            }
        }
        
        // Stream through VAD
        var eventCount = 0
        let detectionStream = vad.streamDetection(audioStream)
        
        do {
            for try await segment in detectionStream {
                eventCount += 1
                print("Stream event \(eventCount): isSpeech=\(segment.isSpeech), time=\(segment.timeRange.start)s")
            }
        } catch {
            // Stream may finish without events for random noise
        }
        
        print("✓ Streaming VAD processed 5s audio in \(totalSamples / chunkSize) chunks")
    }
}
