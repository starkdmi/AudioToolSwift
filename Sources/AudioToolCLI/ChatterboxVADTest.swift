//
//  ChatterboxVADTest.swift
//  ClearVoice
//
//  Minimal test for ChatterBox VAD trim feature - single generation only
//

import Foundation
import ClearVoice
import ClearVoiceCore
@preconcurrency import ClearVoiceTTS
@preconcurrency import AudioUtils
@preconcurrency import MLX

// MARK: - ChatterBox VAD Trim Test (Single Generation)

/// Minimal test to verify VAD trim feature - generates just ONE audio sample
/// to avoid RAM issues from multiple model loads
func runChatterboxVADTest() async throws {
    print("\n=== ChatterBox VAD Trim Test ===")
    print("Single generation test to verify VAD trimming\n")
    
    // Limit GPU memory to 6GB
    GPU.set(memoryLimit: 6 * 1024 * 1024 * 1024)
    
    // Output directory
    let outputDir = FileManager.default.currentDirectoryPath + "/chatterbox_output"
    try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    
    // Reference audio file
    let watsonPath = "/path/to/clear_voice_research/Docs/watson_short.wav"
    guard FileManager.default.fileExists(atPath: watsonPath) else {
        print("❌ Reference file not found: watson_short.wav")
        return
    }
    
    // Create provider with VAD enabled (default)
    print("Loading ChatterBox model (fp32, English, VAD enabled)...")
    let provider = TTSProviders.chatterbox(
        precision: .fp32,
        language: .english,
        useRuAccent: false
    )
    
    let loadStart = Date()
    try await provider.load()
    let loadTime = Date().timeIntervalSince(loadStart)
    print("✓ Model loaded in \(String(format: "%.1f", loadTime))s")
    
    // Set reference audio
    print("Setting reference audio: watson_short.wav")
    try await provider.setReferenceAudio(from: URL(fileURLWithPath: watsonPath))
    print("✓ Reference audio processed\n")
    
    // Generate single test audio WITH VAD trimming (enabled by default)
    let text = "Hello! This is a test of the VAD trimming feature. It should remove breathing sounds and noise from the start and end."
    
    print("Generating WITH VAD trim (enabled by default)...")
    let synthStart = Date()
    let audioWithVAD = try await provider.synthesize(text, voice: "")
    let synthTime = Date().timeIntervalSince(synthStart)
    
    let rtf = audioWithVAD.duration / synthTime
    print("Duration: \(String(format: "%.2f", audioWithVAD.duration))s, RTF: \(String(format: "%.2fx", rtf))")
    
    let outputPath = "\(outputDir)/chatterbox_vad_enabled.wav"
    let saver = AudioSaver(config: .init(sampleRate: Double(audioWithVAD.sampleRate)))
    try saver.save(MLXArray(audioWithVAD.samples), to: outputPath)
    print("✓ Saved: \(outputPath)\n")
    
    // Now disable VAD and generate again for comparison
    print("Disabling VAD trim...")
    await provider.disableVadTrimmer()
    
    print("Generating WITHOUT VAD trim...")
    let synthStart2 = Date()
    let audioNoVAD = try await provider.synthesize(text, voice: "")
    let synthTime2 = Date().timeIntervalSince(synthStart2)
    
    let rtf2 = audioNoVAD.duration / synthTime2
    print("Duration: \(String(format: "%.2f", audioNoVAD.duration))s, RTF: \(String(format: "%.2fx", rtf2))")
    
    let outputPath2 = "\(outputDir)/chatterbox_vad_disabled.wav"
    try saver.save(MLXArray(audioNoVAD.samples), to: outputPath2)
    print("✓ Saved: \(outputPath2)\n")
    
    // Compare
    let durationDiff = audioNoVAD.duration - audioWithVAD.duration
    print("=== VAD Trim Comparison ===")
    print("With VAD:    \(String(format: "%.2f", audioWithVAD.duration))s (\(audioWithVAD.samples.count) samples)")
    print("Without VAD: \(String(format: "%.2f", audioNoVAD.duration))s (\(audioNoVAD.samples.count) samples)")
    print("Trimmed:     \(String(format: "%.2f", durationDiff))s (\(audioNoVAD.samples.count - audioWithVAD.samples.count) samples)")
    
    if durationDiff > 0.05 {
        print("✓ VAD trimming is working - removed \(String(format: "%.2f", durationDiff))s of silence/noise")
    } else {
        print("⚠️ VAD trimming had minimal effect (audio may already be clean)")
    }
    
    // Clean up
    GPU.clearCache()
    
    print("\n=== VAD Test Complete ===")
    print("Compare the two files to hear the difference:")
    print("  - chatterbox_vad_enabled.wav  (trimmed)")
    print("  - chatterbox_vad_disabled.wav (untrimmed)")
}
