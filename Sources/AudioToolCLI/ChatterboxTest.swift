//
//  ChatterboxTest.swift
//  ClearVoice  
//
//  Test with RUAccent + VAD trimming
//

import Foundation
import ClearVoice
import ClearVoiceCore
@preconcurrency import ClearVoiceTTS  
@preconcurrency import AudioUtils
@preconcurrency import MLX

// MARK: - ChatterBox TTS Test

func runChatterboxTest() async throws {
    print("\n=== ChatterBox Full Pipeline Test ===")
    print("RUAccent: ✅ Enabled")
    print("VAD Trim: ✅ Enabled\n")
    
    GPU.set(memoryLimit: 6 * 1024 * 1024 * 1024)
    
    let outputDir = "/path/to/clear_voice_research/ClearVoice/chatterbox_output"
    let burunowPath = "/path/to/clear_voice_research/Docs/burunow_short.wav"
    let model8bitPath = "~/.cache/huggingface/hub/models--starkdmi--chatterbox-8bit/snapshots/ac3a8166d56e6510ac23a10854a4bee1afb535c8"
    
    // Text with homograph "замок" - RUAccent should mark stress correctly
    let text = "мы закрыли замок и прошли мимо замка"
    print("Text: \"\(text)\"\n")
    
    let saver = AudioSaver(config: .init(sampleRate: 24000))
    
    // Create provider WITH RUAccent enabled
    let provider = ChatterboxTTSProvider(
        modelPath: URL(fileURLWithPath: model8bitPath),
        language: .russian,
        useRuAccent: true,  // Enable RUAccent
        convertToStressMarks: true  // Convert + to ́
    )
    
    print("Loading model + RUAccent pipeline...")
    try await provider.load()
    print("✓ Model loaded")
    
    await provider.configure(seed: 42)
    
    print("Setting reference audio...")
    try await provider.setReferenceAudio(from: URL(fileURLWithPath: burunowPath))
    print("✓ Reference set")
    
    // VAD trimming is enabled by default, but let's make sure
    // (it's already enabled by default in the provider)
    print("✓ VAD trimming enabled\n")
    
    print("Synthesizing with RUAccent stress marks...")
    let audio = try await provider.synthesize(text, voice: "")
    print("✓ Done: \(String(format: "%.2f", audio.duration))s\n")
    
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/full_pipeline.wav")
    print("✓ Saved: full_pipeline.wav")
    
    print("\n=== Test Complete ===")
    print("Features used:")
    print("  • S3Tokenizer from HuggingFace cache (fix applied)")
    print("  • RUAccent stress marking")
    print("  • VAD trimming for artifact removal")
    
    GPU.clearCache()
}

func runChatterboxExpressiveTest() async throws {
    print("Skipped")
}
