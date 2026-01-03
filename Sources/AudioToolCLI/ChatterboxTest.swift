//
//  ChatterboxTest.swift
//  ClearVoice  
//
//  Full pipeline test with all fixes
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
    print("S3Tokenizer: ✅ HuggingFace cache (fixed)")
    print("RUAccent: ✅ Enabled")
    print("VAD Trim: ✅ Enabled (threshold 0.7, tuned for TTS)\n")
    
    GPU.set(memoryLimit: 6 * 1024 * 1024 * 1024)
    
    let outputDir = "/path/to/clear_voice_research/ClearVoice/chatterbox_output"
    let burunowPath = "/path/to/clear_voice_research/Docs/burunow_short.wav"
    let model8bitPath = "~/.cache/huggingface/hub/models--starkdmi--chatterbox-8bit/snapshots/ac3a8166d56e6510ac23a10854a4bee1afb535c8"
    
    let text = "мы закрыли замок и прошли мимо замка"
    print("Text: \"\(text)\"\n")
    
    let saver = AudioSaver(config: .init(sampleRate: 24000))
    
    let provider = ChatterboxTTSProvider(
        modelPath: URL(fileURLWithPath: model8bitPath),
        language: .russian,
        useRuAccent: true,
        convertToStressMarks: true
    )
    
    print("Loading model...")
    try await provider.load()
    print("✓ Model loaded")
    
    await provider.configure(seed: 42)
    
    print("Setting reference audio...")
    try await provider.setReferenceAudio(from: URL(fileURLWithPath: burunowPath))
    print("✓ Reference set\n")
    
    print("Synthesizing...")
    let audio = try await provider.synthesize(text, voice: "")
    print("✓ Done: \(String(format: "%.2f", audio.duration))s\n")
    
    try saver.save(MLXArray(audio.samples), to: "\(outputDir)/final_trimmed.wav")
    print("✓ Saved: final_trimmed.wav")
    
    GPU.clearCache()
}

func runChatterboxExpressiveTest() async throws {
    print("Skipped")
}
