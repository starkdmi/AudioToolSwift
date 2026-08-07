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

    private func assertEnhancement(case caseName: String) async throws {
        let artifact = try artifact(caseName)
        let weights = try stagedWeights("MossFormer2_SE_48K_MLX", "model_fp32.safetensors")
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
        let artifact = try artifact("mossformer_gan_se_16k_coreml")
        let model = try reference(
            "Models/python/mossformer_gan_se_coreml/MossFormerGAN_256frames.mlpackage"
        )

        let provider = MossFormerGANCoreMLProvider(modelPath: model.path)
        try await provider.load()

        // One exact segment first: model plus STFT, no stitching. When both this
        // and the full-file case fail, the fault is below the segmenting; when
        // only the full-file case fails, it is the stitching.
        let segment = try XCTUnwrap(artifact.tensor("input_segment"))
        let segmentOutput = try await provider.process(
            AudioBuffer(samples: segment, sampleRate: artifact.sampleRate, channels: 1)
        )
        try expectParity(segmentOutput.samples, matches: "enhanced_segment", in: artifact)

        let fullOutput = try await provider.process(try inputBuffer(artifact))
        try expectParity(fullOutput.samples, matches: "enhanced_full", in: artifact)
    }

    // MARK: - Helpers

    private func inputBuffer(_ artifact: ParityArtifact, channels: Int = 1) throws -> AudioBuffer {
        let samples = try XCTUnwrap(artifact.tensor("input"), "artifact has no 'input' tensor")
        return AudioBuffer(samples: samples, sampleRate: artifact.sampleRate, channels: channels)
    }
}
