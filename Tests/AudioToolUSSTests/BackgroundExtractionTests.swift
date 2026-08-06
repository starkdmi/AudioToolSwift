//
//  BackgroundExtractionTests.swift
//  AudioToolUSSTests
//
//  Test background extraction for GAN SE, USS, and Demucs
//

import XCTest
import AudioToolUSS
import AudioToolCoreML
import AudioToolMLX
import AudioToolCore
import AudioToolTestSupport
import AudioUtils
import MLX

/// Every test here needs real weights and real audio from the sibling research
/// checkout, and none of that exists in a standalone clone.
///
/// This used to reach into `../Models` unconditionally and write its outputs back
/// there - overwriting `frcrn_enhanced.wav`, `demucs_vocals.wav` and five others
/// that belong to those directories, not to this suite. Inputs are now optional
/// and outputs go to a scratch directory.
final class BackgroundExtractionTests: IntegrationTestCase {

    /// Test USS speech separation with background extraction
    func testUSSBackground() async throws {
        let sample = "Models/uss_mlx_swift/USSSwift/Samples/harry_potter_short.wav"
        let testPath = try reference(sample)
        let outputDir = try outputDirectory()

        // Load test audio at 32kHz (USS native rate)
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 32000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testPath)
        eval(audio)
        let samples = audio.asArray(Float.self)

        // Load USS
        let uss = USSProviders.speechSeparation()
        try await uss.load()

        // Separate with background
        let input = AudioBuffer(samples: samples, sampleRate: 32000)
        let result = try await uss.separateWithBackground(input)

        try AudioSaver.saveWAV(MLXArray(result.separated.samples),
                               to: outputDir.appendingPathComponent("uss_speech.wav").path,
                               sampleRate: 32000)
        try AudioSaver.saveWAV(MLXArray(result.background.samples),
                               to: outputDir.appendingPathComponent("uss_background.wav").path,
                               sampleRate: 32000)

        XCTAssertGreaterThan(result.separated.samples.max() ?? 0, 0.01)
    }

    /// Test GAN SE CoreML with background extraction
    func testGANSEBackground() async throws {
        let sample = "Models/mossformer_gan_se_coreml/test.wav"
        let referenceOutput = "Models/mossformer_gan_se_coreml/enhanced_output_no_chunk.wav"
        let model = "Models/mossformer_gan_se_coreml/MossFormerGAN_256frames.mlpackage"
        let testPath = try reference(sample)
        let refPath = try reference(referenceOutput)
        let modelPath = try reference(model)
        let outputDir = try outputDirectory()

        // Load test audio at 16kHz (GAN SE native rate)
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testPath)
        eval(audio)
        let samples = audio.asArray(Float.self)

        // Reference output, for a drift check against the standard path
        let refAudio = try loader.loadMono(from: refPath)
        eval(refAudio)
        let refSamples = refAudio.asArray(Float.self)

        // Load GAN SE (auto-compiles .mlpackage)
        let gan = MossFormerGANCoreMLProvider(modelPath: modelPath.path)
        try await gan.load()

        // Test standard process() first
        let input = AudioBuffer(samples: samples, sampleRate: 16000)
        let enhancedStd = try await gan.process(input)

        // Compare with reference
        let minLen = min(enhancedStd.samples.count, refSamples.count)
        var diff: Float = 0
        for i in 0..<minLen {
            diff += abs(enhancedStd.samples[i] - refSamples[i])
        }
        let avgDiff = diff / Float(minLen)
        XCTAssertLessThan(avgDiff, 0.01, "standard process() drifted from the stored reference output")

        try AudioSaver.saveWAV(MLXArray(enhancedStd.samples),
                               to: outputDir.appendingPathComponent("ganse_enhanced_std.wav").path,
                               sampleRate: 16000)

        // Process with background
        let result = try await gan.processWithBackground(input)

        try AudioSaver.saveWAV(MLXArray(result.enhanced.samples),
                               to: outputDir.appendingPathComponent("ganse_enhanced.wav").path,
                               sampleRate: 16000)
        try AudioSaver.saveWAV(MLXArray(result.background.samples),
                               to: outputDir.appendingPathComponent("ganse_background.wav").path,
                               sampleRate: 16000)

        XCTAssertGreaterThan(result.enhanced.samples.max() ?? 0, 0.01)
    }

    /// Test Demucs vocals/accompaniment separation
    func testDemucsBackground() async throws {
        let sample = "Models/demucs_mlx_swift/test.wav"
        let weights = "Models/demucs_mlx_swift/Weights"
        let testPath = try reference(sample)
        let weightsDir = try reference(weights)
        let outputDir = try outputDirectory()

        // Load test audio at 44.1kHz (Demucs native rate)
        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 44100,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testPath)
        eval(audio)
        let samples = audio.asArray(Float.self)

        // Load Demucs (all 4 source models)
        let demucs = DemucsProvider(weightsDirectory: weightsDir.path)
        try await demucs.loadAll()

        // Separate vocals and accompaniment
        let input = AudioBuffer(samples: samples, sampleRate: 44100)
        let result = try await demucs.separateVocalsWithAccompaniment(input)

        try AudioSaver.saveWAV(MLXArray(result.vocals.samples),
                               to: outputDir.appendingPathComponent("demucs_vocals.wav").path,
                               sampleRate: 44100)
        try AudioSaver.saveWAV(MLXArray(result.accompaniment.samples),
                               to: outputDir.appendingPathComponent("demucs_accompaniment.wav").path,
                               sampleRate: 44100)

        XCTAssertGreaterThan(result.vocals.samples.max() ?? 0, 0.001)
    }

    /// Test FRCRN SE with background extraction
    func testFRCRNBackground() async throws {
        // Same input as the GAN SE test - both models are 16 kHz
        let sample = "Models/mossformer_gan_se_coreml/test.wav"
        let weights = "Models/frcrn_se_mlx_swift/Weights/frcrn_se_16k.safetensors"
        let testPath = try reference(sample)
        let weightsPath = try reference(weights)
        let outputDir = try outputDirectory()

        let loader = AudioLoader(config: AudioLoader.Configuration(
            targetSampleRate: 16000,
            normalizationMode: .none
        ))
        let audio = try loader.loadMono(from: testPath)
        eval(audio)
        let samples = audio.asArray(Float.self)

        // Load FRCRN
        let frcrn = MLXProviders.frcrnSE16K(weightsPath: weightsPath.path)
        try await frcrn.load()

        // Process with background extraction
        let input = AudioBuffer(samples: samples, sampleRate: 16000)
        let result = try await frcrn.processWithBackground(input)

        try AudioSaver.saveWAV(MLXArray(result.enhanced.samples),
                               to: outputDir.appendingPathComponent("frcrn_enhanced.wav").path,
                               sampleRate: 16000)
        try AudioSaver.saveWAV(MLXArray(result.background.samples),
                               to: outputDir.appendingPathComponent("frcrn_background.wav").path,
                               sampleRate: 16000)

        XCTAssertGreaterThan(result.enhanced.samples.max() ?? 0, 0.01)
        XCTAssertGreaterThan(result.background.samples.max() ?? 0, 0.001)
    }
}
