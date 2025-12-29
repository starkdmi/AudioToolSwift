//
//  KokoroTest.swift
//  ClearVoice
//
//  Simple test for Kokoro TTS integration with bf16 model
//

import Foundation
import ClearVoice
import ClearVoiceCore
import ClearVoiceTTS
import AudioUtils
import MLX

// MARK: - Kokoro TTS Test

func runKokoroTest() async throws {
    print("\n=== Kokoro TTS Test (bf16) ===")
    print("Using precision: .bf16 → mlx-community/Kokoro-82M-bf16")
    
    // Create provider with precision-based repo selection
    let tts = TTSProviders.kokoro(
        precision: .bf16,
        language: .americanEnglish
    )
    
    // Observe state for progress
    Task {
        for await state in tts.stateStream {
            switch state {
            case .notLoaded:
                print("State: Not loaded")
            case .downloading(let progress):
                print("State: Downloading \(Int(progress * 100))%")
            case .loading:
                print("State: Loading model...")
            case .ready:
                print("State: Ready ✓")
            case .failed(let error):
                print("State: Failed - \(error)")
            }
        }
    }
    
    // Load (downloads if needed)
    print("\nLoading model...")
    let loadStart = Date()
    try await tts.load()
    print("Model loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")
    
    // Check available voices
    print("\nAvailable voices: \(tts.availableVoices)")
    
    // Load a voice if not auto-loaded
    if tts.availableVoices.isEmpty {
        // Get the model path and load voices
        if let modelPath = ModelDownloader.shared.localPath(for: "mlx-community/Kokoro-82M-bf16") {
            let voicesDir = modelPath.appendingPathComponent("voices")
            print("Looking for voices in: \(voicesDir.path)")
            
            if FileManager.default.fileExists(atPath: voicesDir.path) {
                try tts.loadVoices(from: voicesDir)
                print("Loaded voices: \(tts.availableVoices)")
            } else {
                print("Voices directory not found!")
                return
            }
        }
    }
    
    // Voice naming convention:
    // af_* = American Female, am_* = American Male
    // bf_* = British Female, bm_* = British Male
    // jf_* = Japanese Female, jm_* = Japanese Male (avoid for English)
    
    // Test multiple US/GB voices
    let testVoices = ["af_heart", "af_bella", "am_adam", "bf_emma", "bm_george"]
    let testText = "Hello, this is a test of the Kokoro text to speech system running with ClearVoice."
    
    print("\n--- Testing \(testVoices.count) US/GB voices ---")
    
    var totalSynthTime = 0.0
    var totalDuration = 0.0
    
    for voice in testVoices {
        guard tts.availableVoices.contains(voice) else {
            print("⚠️ Voice '\(voice)' not available, skipping")
            continue
        }
        
        print("\n[\(voice)] Synthesizing...")
        let synthStart = Date()
        let audio = try await tts.synthesize(testText, voice: voice)
        let synthTime = Date().timeIntervalSince(synthStart)
        
        totalSynthTime += synthTime
        totalDuration += audio.duration
        
        let rtf = audio.duration / synthTime
        print("[\(voice)] Duration: \(String(format: "%.2f", audio.duration))s, Synth: \(String(format: "%.2f", synthTime))s, RTF: \(String(format: "%.2fx", rtf))")
        
        // Save each voice
        let outputPath = FileManager.default.currentDirectoryPath + "/kokoro_\(voice).wav"
        let saver = AudioSaver(config: .init(sampleRate: Double(audio.sampleRate)))
        try saver.save(MLXArray(audio.samples), to: outputPath)
        print("[\(voice)] ✓ Saved: \(outputPath)")
    }
    
    // Summary
    let avgRTF = totalDuration / totalSynthTime
    print("\n=== Summary ===")
    print("Voices tested: \(testVoices.count)")
    print("Total audio: \(String(format: "%.2f", totalDuration))s")
    print("Total synth time: \(String(format: "%.2f", totalSynthTime))s")
    print("Average RTF: \(String(format: "%.2fx", avgRTF))")
}
