//
//  VADOnlyTest.swift
//  ClearVoice
//
//  VAD test on harry_potter.wav
//

import Foundation
import AVFoundation
import ClearVoiceCore
@preconcurrency import ClearVoiceFluidAudio

// MARK: - VAD Test

// Compute project root from source file path
private let projectRoot: String = {
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0..<4 { url.deleteLastPathComponent() } // file → Generate → Sources → ClearVoice → ProjectTwo
    return url.path
}()

func runVADOnlyTest() async throws {
    print("\n=== VAD Test on Harry Potter Audio ===\n")
    
    let testFile = "\(projectRoot)/ClearVoice/Docs/harry_potter.wav"
    
    guard FileManager.default.fileExists(atPath: testFile) else {
        print("❌ File not found: \(testFile)")
        return
    }
    
    // Load audio using AVFoundation
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
    
    print("File: harry_potter.wav")
    print("Duration: \(String(format: "%.2f", duration))s")
    print("Original sample rate: \(originalSampleRate)Hz")
    
    // Resample to 16kHz for VAD
    let vadSampleRate = 16000
    let samples = resampleLinear(originalSamples, from: originalSampleRate, to: vadSampleRate)
    let audio = AudioBuffer(samples: samples, sampleRate: vadSampleRate, channels: 1)
    print("Resampled to: \(vadSampleRate)Hz (\(samples.count) samples)\n")
    
    // Test with 0.5 and 0.7 thresholds
    let thresholds: [Float] = [0.5, 0.7]
    
    for threshold in thresholds {
        print("=== Threshold: \(threshold) ===")
        
        let vad = FluidAudioVADProvider(
            threshold: threshold,
            minSpeechDuration: 0.25,
            minSilenceDuration: 0.1
        )
        try await vad.load()
        
        let segments = try await vad.detect(audio)
        
        print("Detected \(segments.count) speech segments:")
        
        var totalSpeech: Double = 0
        for (i, seg) in segments.enumerated() {
            let start = seg.timeRange.start
            let end = seg.timeRange.end
            let segDuration = end - start
            totalSpeech += segDuration
            print("  [\(i+1)] \(String(format: "%.2f", start))s - \(String(format: "%.2f", end))s  (\(String(format: "%.2f", segDuration))s)")
        }
        
        let speechPercent = (totalSpeech / audio.duration) * 100
        print("\nTotal speech: \(String(format: "%.2f", totalSpeech))s / \(String(format: "%.2f", audio.duration))s (\(String(format: "%.0f", speechPercent))%)")
        
        if let first = segments.first {
            print("First speech starts at: \(String(format: "%.2f", first.timeRange.start))s")
        }
        if let last = segments.last {
            print("Last speech ends at: \(String(format: "%.2f", last.timeRange.end))s")
            print("Would trim from end: \(String(format: "%.2f", audio.duration - last.timeRange.end))s")
        }
        print("")
    }
    
    print("=== Test Complete ===")
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
