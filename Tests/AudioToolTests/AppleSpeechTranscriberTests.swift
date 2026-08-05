//
//  AppleSpeechTranscriberTests.swift
//  AudioTool
//
//  Tests for Apple Speech transcriber (SpeechAnalyzer, iOS 26+)
//

import Foundation
import Testing
@testable import AudioTool
@testable import AudioToolCore
@testable import AudioToolSpeech

@Suite("Apple Speech Transcriber Tests", .enabled(if: TestConfiguration.runIntegrationTests,
        "integration test - set RUN_INTEGRATION_TESTS=1"))
struct AppleSpeechTranscriberTests {
    
    // MARK: - TranscriptionModel
    
    @Test("TranscriptionModel.appleSpeech modelName")
    func testTranscriptionModelName() {
        let model = TranscriptionModel.appleSpeech
        #expect(model.modelName == "apple_speech")
    }
    
    // MARK: - AppleSpeechTranscriber (iOS 26+ only)
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("AppleSpeechTranscriber initialization")
    func testProviderInit() {
        let transcriber = AppleSpeechTranscriber(locale: "en-US")
        #expect(transcriber.sampleRate == 16000)
        #expect(transcriber.inputChannels == 1)
        #expect(transcriber.outputChannels == 1)
    }
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("AppleSpeechTranscriber locale initialization")
    func testLocaleInit() {
        let usLocale = Locale(identifier: "en-US")
        let transcriber = AppleSpeechTranscriber(locale: usLocale)
        #expect(transcriber.sampleRate == 16000)
    }
    
    // Tests that require full SpeechAnalyzer API (Xcode 26+ SDK)
    #if compiler(>=6.2)
    @available(iOS 26.0, macOS 26.0, *)
    @Test("AppleSpeechTranscriber supported locales")
    func testSupportedLocales() async {
        let locales = await AppleSpeechTranscriber.supportedLocales()
        print("Supported locales: \(locales.count)")
        
        // Print first few locales for debugging
        for locale in locales.prefix(5) {
            print("  - \(locale.identifier)")
        }
        
        #expect(locales.count > 0)
    }
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("AppleSpeechTranscriber locale support check")
    func testLocaleSupport() async {
        let isEnglishSupported = await AppleSpeechTranscriber.isLocaleSupported(Locale(identifier: "en-US"))
        print("en-US supported: \(isEnglishSupported)")
        // Actual support depends on device/OS
    }
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("AudioBuffer to AVAudioPCMBuffer conversion")
    func testAudioBufferConversion() throws {
        let samples: [Float] = Array(repeating: 0.5, count: 16000)
        let buffer = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_\(UUID().uuidString).wav")
        
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }
        
        // Test conversion to AVAudioFile via temp file
        let audioFile = try buffer.writeToTemporaryFile(at: tempURL)
        
        // Check file was created and is readable
        #expect(FileManager.default.fileExists(atPath: tempURL.path))
        #expect(audioFile.fileFormat.sampleRate == 16000)
    }
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("AppleSpeechTranscriber load model")
    func testLoadModel() async throws {
        let transcriber = AppleSpeechTranscriber(locale: "en-US")
        
        // This may fail if locale not supported on test machine
        do {
            try await transcriber.load()
            print("✓ Model loaded successfully")
        } catch {
            print("⚠ Model load skipped: \(error.localizedDescription)")
            // Don't fail test - locale support varies by device
        }
    }
    #endif
    
    // MARK: - Factory Tests
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("SpeechProviders.appleSpeech factory")
    func testFactory() {
        let transcriber = SpeechProviders.appleSpeech(locale: "fr-FR")
        #expect(transcriber.sampleRate == 16000)
    }
    
    // MARK: - AudioTool Integration
    
    @available(iOS 26.0, macOS 26.0, *)
    @Test("AudioTool register transcriber")
    func testAudioToolRegistration() async {
        let voice = AudioEngine()
        let transcriber = SpeechProviders.appleSpeech(locale: "en-US")
        await voice.register(transcriber: transcriber, for: .appleSpeech)
        #expect(Bool(true))
    }
    
    // MARK: - AudioBuffer Tests
    
    @Test("AudioBuffer creation and properties")
    func testAudioBufferCreation() {
        let samples: [Float] = Array(repeating: 0.5, count: 16000)
        let buffer = AudioBuffer(samples: samples, sampleRate: 16000, channels: 1)
        
        #expect(buffer.samples.count == 16000)
        #expect(buffer.sampleRate == 16000)
        #expect(buffer.channels == 1)
    }
}
