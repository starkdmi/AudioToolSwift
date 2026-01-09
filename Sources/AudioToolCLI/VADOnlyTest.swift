//
//  VADOnlyTest.swift
//  ClearVoice
//
//  VAD parameter tuning test
//

import Foundation
import AVFoundation
import ClearVoiceCore
@preconcurrency import ClearVoiceFluidAudio

// MARK: - VAD Parameter Tuning Test

func runVADOnlyTest() async throws {
    print("\n=== VAD Parameter Tuning Test ===")
    print("Testing minSilenceDuration and padding on chatterbox_vad_disabled.wav\n")
    
    let testFile = "/path/to/clear_voice_research/ClearVoice/chatterbox_output/chatterbox_vad_disabled.wav"
    
    guard FileManager.default.fileExists(atPath: testFile) else {
        print("❌ File not found")
        return
    }
    
    // Load audio using AVFoundation at 16kHz for VAD
    let url = URL(fileURLWithPath: testFile)
    guard let audioFile = try? AVAudioFile(forReading: url) else {
        print("❌ Could not load audio file")
        return
    }
    
    let format = audioFile.processingFormat
    let frameCount = UInt32(audioFile.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        print("❌ Could not create buffer")
        return
    }
    
    try audioFile.read(into: buffer)
    
    guard let floatData = buffer.floatChannelData?[0] else {
        print("❌ Could not get float data")
        return
    }
    
    let originalSamples = Array(UnsafeBufferPointer(start: floatData, count: Int(buffer.frameLength)))
    let originalSampleRate = Int(format.sampleRate)
    let duration = Double(originalSamples.count) / Double(originalSampleRate)
    
    print("File: chatterbox_vad_disabled.wav")
    print("Duration: \(duration)s")
    print("Original sample rate: \(originalSampleRate)Hz")
    
    // Resample to 16kHz for VAD
    let vadSampleRate = 16000
    let samples = resampleLinear(originalSamples, from: originalSampleRate, to: vadSampleRate)
    let audio = AudioBuffer(samples: samples, sampleRate: vadSampleRate, channels: 1)
    print("Resampled to: \(vadSampleRate)Hz (\(samples.count) samples)\n")
    
    // Test grid: thresholds x minSilence x padding
    let thresholds: [Float] = [0.5, 0.7, 0.9, 0.95]
    let minSilences: [Double] = [0.1, 0.05, 0.03]  // Python default is 0.1
    let paddings: [Double] = [0.03, 0.01, 0.0]
    
    print("Threshold | MinSilence | Padding | SpeechEnd | TrimAmount")
    print("----------|------------|---------|-----------|----------")
    
    for threshold in thresholds {
        for minSilence in minSilences {
            let vad = FluidAudioVADProvider(
                threshold: threshold,
                minSpeechDuration: 0.1,
                minSilenceDuration: minSilence
            )
            try await vad.load()
            
            let segments = try await vad.detect(audio)
            
            if let last = segments.last {
                for padding in paddings {
                    let endWithPad = min(last.timeRange.end + padding, audio.duration)
                    let trimAmount = audio.duration - endWithPad
                    print("\(threshold)       | \(minSilence)       | \(padding)    | \(String(format: "%.2f", endWithPad))s     | \(String(format: "%.2f", trimAmount))s")
                }
            } else {
                print("\(threshold)       | \(minSilence)       | *       | NO SPEECH | -")
            }
        }
    }
    
    print("\n=== Test Complete ===")
}

/// Linear interpolation resampling
func resampleLinear(_ samples: [Float], from srcRate: Int, to dstRate: Int) -> [Float] {
    guard srcRate != dstRate else { return samples }
    
    let ratio = Double(dstRate) / Double(srcRate)
    let newCount = Int(Double(samples.count) * ratio)
    var resampled = [Float](repeating: 0, count: newCount)
    
    for i in 0..<newCount {
        let srcPos = Double(i) / ratio
        let srcIdx = Int(srcPos)
        let frac = Float(srcPos - Double(srcIdx))
        
        if srcIdx + 1 < samples.count {
            resampled[i] = samples[srcIdx] * (1 - frac) + samples[srcIdx + 1] * frac
        } else if srcIdx < samples.count {
            resampled[i] = samples[srcIdx]
        }
    }
    
    return resampled
}
