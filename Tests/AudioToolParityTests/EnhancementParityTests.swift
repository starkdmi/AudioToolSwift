//
//  EnhancementParityTests.swift
//  AudioToolParityTests
//
//  FRCRN, MossFormer2 SE and the CoreML GAN against their MLX Python references.
//

import AudioToolCore
import AudioToolCoreML
import AudioToolMLX
import AudioToolTestSupport
import XCTest

final class EnhancementParityTests: ParityTestCase {

    /// FRCRN 16 kHz, whole-file.
    ///
    /// Weights come from the same file the artifact's sidecar hashes, so a
    /// mismatch here cannot be the two sides running different checkpoints.
    func testFRCRNMatchesReference() async throws {
        let artifact = try artifact("frcrn_se_16k")
        let weights = try reference("Models/python/frcrn_se_mlx/frcrn_se_16k.safetensors")
        let input = try inputBuffer(artifact)

        let provider = FRCRNSE16KProvider(weightsPath: weights.path)
        try await provider.load()
        let output = try await provider.process(input)

        try expectParity(output.samples, matches: "enhanced", in: artifact)
    }

    /// MossFormer2 SE 48 kHz, fp32, chunked - the path anything over four seconds
    /// takes. The reference is chunked identically via benchmark_chunking.py.
    ///
    /// The input is already at the model's rate, so no resampler is inside this
    /// comparison - the reference's own resample branch does not run either.
    func testMossFormer2SEMatchesReference() async throws {
        try await assertEnhancement(case: "mossformer2_se_48k")
    }

    /// The same model on a 3-second clip, under the provider's 4-second
    /// `maxDirectDuration`, so neither side chunks. Isolates the model from the
    /// chunking wrapper.
    func testMossFormer2SEDirectMatchesReference() async throws {
        try await assertEnhancement(case: "mossformer2_se_48k_direct")
    }

    /// The quantized checkpoints, direct, one test per bit width.
    ///
    /// These are held to the same round-off bounds as fp32, not looser ones, and
    /// that is the point. Both sides load the *same packed bytes* and dequantize
    /// through the same MLX kernels, so quantization error is common to both and
    /// cancels: what is left is only the port difference. A loosened threshold here
    /// would accept exactly the failures worth catching - a group size read from
    /// the wrong place, a different set of layers quantized, a dequantization that
    /// disagrees between runtimes. Each of those still yields Swift output that
    /// degrades smoothly against Swift fp32, so only the cross-language comparison
    /// sees them.
    ///
    /// How much quality each precision actually costs is a different question, and
    /// not one parity answers - see the trade-off report.
    func testMossFormer2SEDirectFP16MatchesReference() async throws {
        try await assertEnhancement(case: "mossformer2_se_48k_direct_fp16", weights: "model_fp16.safetensors")
    }

    func testMossFormer2SEDirectInt8MatchesReference() async throws {
        try await assertEnhancement(case: "mossformer2_se_48k_direct_int8", weights: "model_int8.safetensors")
    }

    func testMossFormer2SEDirectInt6MatchesReference() async throws {
        try await assertEnhancement(case: "mossformer2_se_48k_direct_int6", weights: "model_int6.safetensors")
    }

    func testMossFormer2SEDirectInt4MatchesReference() async throws {
        try await assertEnhancement(case: "mossformer2_se_48k_direct_int4", weights: "model_int4.safetensors")
    }

    private func assertEnhancement(
        case caseName: String,
        weights weightsFile: String = "model_fp32.safetensors"
    ) async throws {
        let artifact = try artifact(caseName)
        let weights = try stagedWeights("MossFormer2_SE_48K_MLX", weightsFile)
        let input = try inputBuffer(artifact)

        let provider = MossFormer2SE48KProvider(weightsPath: weights.path)
        try await provider.load()
        let output = try await provider.process(input)

        try expectParity(output.samples, matches: "enhanced", in: artifact)
    }

    /// MossFormerGAN, CoreML.
    ///
    /// The tightest expected bound of the set: both sides run the *same* compiled
    /// `.mlpackage` and both do STFT/ISTFT in MLX, so the only things that can
    /// differ are framing, normalisation and segment stitching - Swift wrapper
    /// code, all of it touched recently. A low number here is a wrapper bug, not
    /// the CoreML conversion showing through.
    func testMossFormerGANMatchesReference() async throws {
        try await assertGAN(model: "MossFormerGAN_256frames.mlpackage", prefix: "")
    }

    /// The FP16 conversion of the same graph, against the same fp32 reference.
    ///
    /// Unlike every other precision case here, this one is *not* held to the fp32
    /// bound and cannot be: there is no fp16 reference to compare against, because
    /// the Python side runs the MLX implementation rather than a CoreML conversion.
    /// So this measures conversion loss and port error together, and the number is
    /// legitimately lower.
    ///
    /// It exists because the alternative was reporting a speed result on unverified
    /// output. The FP16 package returns `MLMultiArrayDataTypeFloat16`, which
    /// `parseModelOutputToMLX` rejected outright until it learned the type - and the
    /// de-striding it then has to do is exactly the step that, done wrong, produced
    /// a sheared spectrogram measuring -1.1 dB while looking like a working model.
    /// A benchmark would have reported that as a 14% speedup.
    func testMossFormerGANFP16MatchesReference() async throws {
        try await assertGAN(model: "MossFormerGAN_256frames_FP16.mlpackage", prefix: "fp16_")
    }

    private func assertGAN(model modelName: String, prefix: String) async throws {
        let artifact = try artifact("mossformer_gan_se_16k_coreml")
        let model = try reference("Models/python/mossformer_gan_se_coreml/\(modelName)")

        let provider = MossFormerGANCoreMLProvider(modelPath: model.path)
        try await provider.load()

        // One exact segment first: model plus STFT, no stitching. When both this
        // and the full-file case fail, the fault is below the segmenting; when
        // only the full-file case fails, it is the stitching.
        let segment = try XCTUnwrap(artifact.tensor("input_segment"))
        let segmentOutput = try await provider.process(
            AudioBuffer(samples: segment, sampleRate: artifact.sampleRate, channels: 1)
        )
        try expectParity(
            segmentOutput.samples, matches: "enhanced_segment", in: artifact, keyPrefix: prefix
        )

        let fullOutput = try await provider.process(try inputBuffer(artifact))
        try expectParity(
            fullOutput.samples, matches: "enhanced_full", in: artifact, keyPrefix: prefix
        )
    }

    // MARK: - Helpers

    private func inputBuffer(_ artifact: ParityArtifact, channels: Int = 1) throws -> AudioBuffer {
        let samples = try XCTUnwrap(artifact.tensor("input"), "artifact has no 'input' tensor")
        return AudioBuffer(samples: samples, sampleRate: artifact.sampleRate, channels: channels)
    }
}
