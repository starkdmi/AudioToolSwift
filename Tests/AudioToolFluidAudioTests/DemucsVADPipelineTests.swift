//
//  DemucsVADPipelineTests.swift
//  AudioToolFluidAudioTests
//
//  Tests for Demucs → VAD pipeline on music/singing
//

import XCTest
import AudioToolTestSupport
import AudioToolFluidAudio
import AudioToolMLX
import AudioToolCore
import AudioUtils
import MLX

final class DemucsVADPipelineTests: IntegrationTestCase {
    
    // Compute project root from source file path
    
    /// Test Demucs → VAD pipeline on all three test files
    func testDemucsVADPipeline() async throws {
        print("\n=== Demucs → VAD Pipeline Test ===")
        print("Testing if vocal separation improves VAD detection\n")
        
        // Get Demucs weights from local Models directory
        let weightsPath = try reference("Models/demucs_mlx_swift/Weights").path
        
        // Verify weights exist
        guard FileManager.default.fileExists(atPath: weightsPath + "/vocals.safetensors") else {
            throw XCTSkip("Demucs weights not found at \(weightsPath)")
        }
        
        print("Loading Demucs from: \(weightsPath)")
        
        let demucs = MLXProviders.demucs(weightsDirectory: weightsPath)
        try await demucs.load(stem: .vocals)
        print("  Demucs loaded for vocals\n")
        
        // Test files
        let testFiles = ["billions", "music_35s", "watson_30s"]
        
        for filename in testFiles {
            guard let testURL = Bundle.module.url(forResource: filename, withExtension: "wav", subdirectory: "Fixtures") else {
                print("⚠️ \(filename).wav not found, skipping")
                continue
            }
            
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("Testing: \(filename).wav")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            try await testFileWithDemucsVAD(url: testURL, filename: filename, demucs: demucs)
            print()
        }
        
        print("=== Pipeline Test Complete ===")
    }
    
    private func testFileWithDemucsVAD(url: URL, filename: String, demucs: DemucsProvider) async throws {
        // 1. Load audio at 44.1kHz for Demucs
        let loader44k = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 44100,
            normalizationMode: .none
        ))
        let audio44k = try loader44k.loadMono(from: url)
        eval(audio44k)
        let samples44k = audio44k.asArray(Float.self)
        
        let duration = Double(samples44k.count) / 44100.0
        print("\nAudio: \(String(format: "%.1f", duration))s at 44.1kHz")
        
        // 2. Run Demucs to separate vocals
        print("Running Demucs separation...")
        let startDemucs = Date()
        
        let input44k = AudioBuffer(samples: samples44k, sampleRate: 44100, channels: 1)
        let vocalsBuffer = try await demucs.separate(input44k, stem: .vocals)
        
        let demucsTime = Date().timeIntervalSince(startDemucs)
        print("  Demucs completed in \(String(format: "%.2f", demucsTime))s (RTF: \(String(format: "%.1f", duration/demucsTime))x)")
        
        let vocalsSamples = vocalsBuffer.samples
        print("  Vocals extracted: \(vocalsSamples.count) samples")
        
        // Check vocals level
        let vocalsMax = vocalsSamples.map { abs($0) }.max() ?? 0
        let vocalsRMS = sqrt(vocalsSamples.map { $0 * $0 }.reduce(0, +) / Float(vocalsSamples.count))
        print("  Vocals levels: max=\(String(format: "%.4f", vocalsMax)), RMS=\(String(format: "%.4f", vocalsRMS))")
        
        // 3. Resample vocals to 16kHz for VAD
        // Simple downsampling (not ideal but works for testing)
        let downsampleFactor = 44100.0 / 16000.0
        var vocals16k = [Float]()
        for i in stride(from: 0.0, to: Double(vocalsSamples.count), by: downsampleFactor) {
            let idx = Int(i)
            if idx < vocalsSamples.count {
                vocals16k.append(vocalsSamples[idx])
            }
        }
        
        print("  Resampled to 16kHz: \(vocals16k.count) samples")
        
        // 4. Run VAD on isolated vocals
        print("\nRunning VAD on isolated vocals...")
        
        let vad = FluidAudioProviders.sileroVAD(threshold: 0.5)  // Default threshold should work better on clean vocals
        try await vad.load()
        
        let vocalsInput = AudioBuffer(samples: vocals16k, sampleRate: 16000, channels: 1)
        let startVAD = Date()
        let segments = try await vad.detect(vocalsInput)
        let vadTime = Date().timeIntervalSince(startVAD)
        
        let totalSpeech = segments.reduce(0.0) { $0 + ($1.timeRange.end - $1.timeRange.start) }
        let vocalsDuration = Double(vocals16k.count) / 16000.0
        
        print("  VAD completed in \(String(format: "%.3f", vadTime))s")
        print("  Detected \(segments.count) segments, total \(String(format: "%.1f", totalSpeech))s (\(String(format: "%.0f", totalSpeech/vocalsDuration*100))%)")
        
        for (i, seg) in segments.enumerated() {
            print("    [\(i+1)] \(String(format: "%.2f", seg.timeRange.start))s - \(String(format: "%.2f", seg.timeRange.end))s")
        }
        
        // 5. Compare with direct VAD (no Demucs)
        print("\nComparing with direct VAD (no Demucs)...")
        
        // Load at 16kHz for direct VAD
        let loader16k = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio16k = try loader16k.loadMono(from: url)
        eval(audio16k)
        
        let directVAD = FluidAudioProviders.sileroVAD(threshold: 0.5)
        try await directVAD.load()
        
        let directInput = AudioBuffer(samples: audio16k.asArray(Float.self), sampleRate: 16000, channels: 1)
        let directSegments = try await directVAD.detect(directInput)
        let directTotal = directSegments.reduce(0.0) { $0 + ($1.timeRange.end - $1.timeRange.start) }
        
        print("  Direct VAD: \(directSegments.count) segments, \(String(format: "%.1f", directTotal))s (\(String(format: "%.0f", directTotal/duration*100))%)")
        
        // Summary
        print("\n📊 Summary for \(filename):")
        print("  Direct VAD (0.5):     \(directSegments.count) segments, \(String(format: "%.1f", directTotal))s")
        print("  Demucs→VAD (0.5):     \(segments.count) segments, \(String(format: "%.1f", totalSpeech))s")
        
        if totalSpeech > directTotal {
            print("  ✓ Demucs improved detection by \(String(format: "%.0f", (totalSpeech - directTotal) / directTotal * 100))%")
        } else if directTotal > 0 && totalSpeech < directTotal {
            print("  ⚠️ Demucs detected \(String(format: "%.0f", (directTotal - totalSpeech) / directTotal * 100))% less")
        }
    }
}
