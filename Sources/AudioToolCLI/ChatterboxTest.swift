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

// Compute project root from source file path
private let chatterboxProjectRoot: String = {
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0..<4 { url.deleteLastPathComponent() }
    return url.path
}()

func runChatterboxTest() async throws {
    print("\n=== ChatterBox Full Pipeline Test ===")
    print("S3Tokenizer: ✅ HuggingFace cache (fixed)")
    print("RUAccent: ✅ Enabled")
    print("VAD Trim: ✅ Enabled (threshold 0.7, tuned for TTS)\n")
    
    GPU.set(memoryLimit: 6 * 1024 * 1024 * 1024)
    
    let outputDir = "\(chatterboxProjectRoot)/ClearVoice/chatterbox_output"
    let burunowPath = "\(chatterboxProjectRoot)/ClearVoice/Docs/burunow_short.wav"
    // HuggingFace cache is user-specific, use home directory
    let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
    let modelPath = "\(homeDir)/.cache/huggingface/hub/models--starkdmi--chatterbox-fp16/snapshots/e68aaba8ef36cb9a8a8c9a5807c1b1004b113c70"
    
    let text = "мы закрыли замок и прошли мимо замка"
    print("Text: \"\(text)\"\n")
    
    let saver = AudioSaver(config: .init(sampleRate: 24000))
    
    let provider = ChatterboxTTSProvider(
        modelPath: URL(fileURLWithPath: modelPath),
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
