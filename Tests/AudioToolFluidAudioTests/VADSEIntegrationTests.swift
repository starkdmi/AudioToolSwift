//
//  VADSEIntegrationTests.swift
//  AudioToolFluidAudioTests
//
//  Full integration test: VAD → SE 48K pipeline with real audio
//

import XCTest
import AudioToolTestSupport
import AudioToolFluidAudio
import AudioToolMLX
import AudioToolCore
import AudioUtils
import MLX

final class VADSEIntegrationTests: IntegrationTestCase {
    
    /// Full pipeline test: VAD detects speech → SE 48K enhances speech segments
    func testVADToSE48KPipeline() async throws {
        print("\n=== Full VAD → SE 48K Pipeline Test ===")
        
        // 1. Setup providers
        print("\n1. Loading models...")
        let startLoad = Date()
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        print("   VAD loaded")
        
        let se = MLXProviders.mossformer2SE48K()
        try await se.load()
        print("   SE 48K loaded")
        
        let loadTime = Date().timeIntervalSince(startLoad)
        print("   Total load time: \(String(format: "%.2f", loadTime))s")
        
        // 2. Load test audio at 16kHz (for VAD)
        print("\n2. Loading test audio...")
        guard let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("Test fixture test.wav not found")
        }
        
        let loader16k = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio16k = try loader16k.loadMono(from: testURL)
        eval(audio16k)
        let samples16k = audio16k.asArray(Float.self)
        
        let audioDuration = Double(samples16k.count) / 16000.0
        print("   Loaded: \(samples16k.count) samples at 16kHz (\(String(format: "%.2f", audioDuration))s)")
        
        // 3. Run VAD
        print("\n3. Running VAD...")
        let startVAD = Date()
        
        let vadInput = AudioBuffer(samples: samples16k, sampleRate: 16000, channels: 1)
        let segments = try await vad.detect(vadInput)
        
        let vadTime = Date().timeIntervalSince(startVAD)
        let vadRTF = audioDuration / vadTime
        
        print("   VAD completed in \(String(format: "%.3f", vadTime))s (RTF: \(String(format: "%.0f", vadRTF))x)")
        print("   Detected \(segments.count) speech segments:")
        
        for (i, segment) in segments.enumerated() {
            let segDuration = segment.timeRange.end - segment.timeRange.start
            print("     [\(i+1)] \(String(format: "%.2f", segment.timeRange.start))s - \(String(format: "%.2f", segment.timeRange.end))s (\(String(format: "%.1f", segDuration))s)")
        }
        
        // 4. Extract and enhance speech segments
        print("\n4. Enhancing speech segments with SE 48K...")
        
        // Load audio at 48kHz for SE
        let loader48k = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 48000,
            normalizationMode: .none
        ))
        let audio48k = try loader48k.loadMono(from: testURL)
        eval(audio48k)
        let samples48k = audio48k.asArray(Float.self)
        
        var enhancedSegments: [AudioBuffer] = []
        var totalSpeechDuration = 0.0
        var totalEnhanceTime = 0.0
        
        for (i, segment) in segments.enumerated() {
            // Convert time to 48kHz sample indices
            let startSample = Int(segment.timeRange.start * 48000.0)
            let endSample = min(Int(segment.timeRange.end * 48000.0), samples48k.count)
            
            guard endSample > startSample else { continue }
            
            let chunkSamples = Array(samples48k[startSample..<endSample])
            let chunk = AudioBuffer(samples: chunkSamples, sampleRate: 48000, channels: 1)
            
            let startEnhance = Date()
            let enhanced = try await se.process(chunk)
            let enhanceTime = Date().timeIntervalSince(startEnhance)
            
            enhancedSegments.append(enhanced)
            totalSpeechDuration += chunk.duration
            totalEnhanceTime += enhanceTime
            
            print("   Segment \(i+1): enhanced \(String(format: "%.1f", chunk.duration))s in \(String(format: "%.3f", enhanceTime))s")
        }
        
        // 5. Summary
        print("\n=== Pipeline Results ===")
        print("   Input audio: \(String(format: "%.2f", audioDuration))s")
        print("   Speech segments: \(segments.count)")
        print("   Total speech: \(String(format: "%.2f", totalSpeechDuration))s (\(String(format: "%.0f", totalSpeechDuration/audioDuration*100))% of audio)")
        print("   VAD time: \(String(format: "%.3f", vadTime))s (RTF: \(String(format: "%.0f", vadRTF))x)")
        print("   SE time: \(String(format: "%.3f", totalEnhanceTime))s (RTF: \(String(format: "%.1f", totalSpeechDuration/totalEnhanceTime))x)")
        print("   Total pipeline time: \(String(format: "%.3f", vadTime + totalEnhanceTime))s")
        
        let overallRTF = audioDuration / (vadTime + totalEnhanceTime)
        print("   Overall RTF: \(String(format: "%.1f", overallRTF))x")
        
        // 6. Assertions - use CI-aware thresholds
        let isCI = ProcessInfo.processInfo.environment["CI"] == "1"
        let vadRTFThreshold = isCI ? 20.0 : 100.0
        
        XCTAssertFalse(segments.isEmpty, "Should detect at least one speech segment")
        XCTAssertEqual(enhancedSegments.count, segments.count, "Should enhance all segments")
        XCTAssertGreaterThan(vadRTF, vadRTFThreshold, "VAD should be >\(vadRTFThreshold)x real-time")
        
        print("\n✓ VAD → SE 48K pipeline test passed!")
    }
    
    /// Test processing only the first segment (for speed)
    func testVADSingleSegmentEnhancement() async throws {
        print("\n=== VAD → Single Segment SE Enhancement ===")
        
        // Load providers
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await vad.load()
        
        let se = MLXProviders.mossformer2SE48K()
        try await se.load()
        
        // Load test audio
        guard let testURL = Bundle.module.url(forResource: "test", withExtension: "wav", subdirectory: "Fixtures") else {
            throw XCTSkip("Test fixture not found")
        }
        
        // VAD at 16kHz
        let loader16k = AudioLoader(config: AudioLoader.Configuration(targetSampleRate: 16000))
        let audio16k = try loader16k.loadMono(from: testURL)
        eval(audio16k)
        
        let vadInput = AudioBuffer(samples: audio16k.asArray(Float.self), sampleRate: 16000, channels: 1)
        let segments = try await vad.detect(vadInput)
        
        guard let firstSegment = segments.first else {
            print("No speech detected - audio may be too noisy or too quiet")
            return
        }
        
        print("First speech segment: \(String(format: "%.2f", firstSegment.timeRange.start))s - \(String(format: "%.2f", firstSegment.timeRange.end))s")
        
        // Enhance first segment at 48kHz
        let loader48k = AudioLoader(config: AudioLoader.Configuration(targetSampleRate: 48000))
        let audio48k = try loader48k.loadMono(from: testURL)
        eval(audio48k)
        let samples48k = audio48k.asArray(Float.self)
        
        let startSample = Int(firstSegment.timeRange.start * 48000.0)
        let endSample = min(Int(firstSegment.timeRange.end * 48000.0), samples48k.count)
        
        let chunkSamples = Array(samples48k[startSample..<endSample])
        let chunk = AudioBuffer(samples: chunkSamples, sampleRate: 48000, channels: 1)
        
        let enhanced = try await se.process(chunk)
        
        XCTAssertEqual(enhanced.samples.count, chunk.samples.count, "Enhanced should match input length")
        print("✓ Enhanced first speech segment: \(enhanced.samples.count) samples")
    }
}
