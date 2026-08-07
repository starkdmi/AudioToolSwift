//
//  UpscalingParityTests.swift
//  AudioToolParityTests
//
//  MossFormer2 SR and USS.
//

import AudioToolCore
@testable import AudioToolMLX
import AudioToolTestSupport
import AudioUtils
import MLX
import AudioToolUSS
import USSMLXSwift
import XCTest

final class UpscalingParityTests: ParityTestCase {

    /// MossFormer2 SR, 16 kHz in -> 48 kHz out.
    ///
    /// This is the one comparison with a resampler inside it. The reference
    /// upsamples 16 -> 48 kHz with `librosa.resample` before the mel front-end, so
    /// `enhanced` folds the resampler and the model into one number, and a
    /// resampler difference is indistinguishable from a model difference here.
    ///
    /// That is why the artifact also carries `upsampled_48k`, librosa's output on
    /// its own. Feeding the provider a pre-upsampled tensor would separate the two
    /// cleanly, but there is no entry point for it today - the provider resamples
    /// internally. Worth adding one; until then this number is the composite and
    /// `upsampled_48k` is sitting there ready for a test of the seam alone.
    func testSuperResolutionMatchesReference() async throws {
        try await assertUpscaling(case: "mossformer2_sr_48k")
    }

    /// A 3-second clip, under the 4-second `maxDirectDuration`, so neither side
    /// chunks. Still contains both resamplers, so a gap between this and the
    /// chunked case separates the seams from the resampler.
    func testSuperResolutionDirectMatchesReference() async throws {
        try await assertUpscaling(case: "mossformer2_sr_48k_direct")
    }

    private func assertUpscaling(case caseName: String) async throws {
        let artifact = try artifact(caseName)
        let weights = try stagedWeights("MossFormer2_SR_48K_MLX", "model_fp32.safetensors")
        let config = try stagedWeights("MossFormer2_SR_48K_MLX", "config.json")
        let samples = try XCTUnwrap(artifact.tensor("input"))
        let inputRate = 16_000

        let provider = MossFormer2SR48KProvider(weightsPath: weights.path, configPath: config.path)
        try await provider.load()
        let output = try await provider.process(
            AudioBuffer(samples: samples, sampleRate: inputRate, channels: 1)
        )

        XCTAssertEqual(
            output.sampleRate, artifact.sampleRate,
            "SR should return \(artifact.sampleRate) Hz"
        )
        try expectParity(output.samples, matches: "enhanced", in: artifact)
    }

    /// Isolate the 16 -> 48 kHz upsample, which is the first step of SR inference.
    ///
    /// SR is the only case that stayed wrong after its reference was chunked to
    /// match - and it is wrong in the *direct* case too, where nothing chunks. So
    /// the fault is upstream of the model. This reproduces
    /// `MLXSuperResolutionProvider.resampleTo48k` exactly - float32 wav out,
    /// AudioLoader back in at 48 kHz - and compares it against librosa's output,
    /// which the artifact carries for this purpose.
    func testSuperResolutionResamplerMatchesReference() async throws {
        let artifact = try artifact("mossformer2_sr_48k_direct")
        let input = try XCTUnwrap(artifact.tensor("input"))
        let reference = try XCTUnwrap(artifact.tensor("upsampled_48k"))

        // The old path, reproduced: float32 wav out, AudioLoader back in. Kept so
        // the improvement is measured rather than asserted.
        let directory = try TestGate.outputDirectory("\(Self.self)/resampler")
        let path = directory.appendingPathComponent("sr_input.wav")
        try AudioSaver(config: .init(sampleRate: 16_000)).save(MLXArray(input), to: path)
        let loader = AudioLoader(config: .init(targetSampleRate: 48_000, normalizationMode: .none))
        let viaFile = try loader.loadMono(from: path)
        eval(viaFile)

        let inMemory = try SuperResolutionResampler.upsample(input, from: 16_000, to: 48_000)

        print(String(format: "DIAG expected 16k->48k: %d * 3 = %d", input.count, input.count * 3))
        for (label, candidate) in [("wav round-trip", viaFile.asArray(Float.self)),
                                   ("in-memory", inMemory)] {
            let overlap = min(reference.count, candidate.count)
            let snr = ParityMetrics.snrDB(
                reference: Array(reference.prefix(overlap)),
                candidate: Array(candidate.prefix(overlap))
            )
            print(String(
                format: "DIAG   %-16@ %6d samples  first %d agree at %.1f dB",
                label as NSString, candidate.count, overlap, snr
            ))
        }

        XCTAssertEqual(
            inMemory.count, input.count * 3,
            "an integer-ratio upsample must be exact - this is the 2688-sample tail loss"
        )
    }

    /// USS, query-conditioned, at fp32.
    ///
    /// `useFp16: false` is not incidental - the artifact was generated from
    /// `resunet30_fp32.safetensors`, and the provider defaults to fp16. Comparing
    /// an fp16 run against an fp32 reference would produce a number that looks
    /// like a port bug and is not one.
    ///
    /// Two conditions against one mixture. If the port ever dropped or mis-shaped
    /// the 527-d embedding, both outputs would agree with each other while
    /// disagreeing with the reference - which a single-condition test would miss.
    func testUniversalSourceSeparationMatchesReference() async throws {
        let artifact = try artifact("uss_resunet30_32k")
        let weights = try reference(
            "Models/uss_mlx_swift/USSSwift/Models/resunet30_fp32.safetensors"
        )
        let samples = try XCTUnwrap(artifact.tensor("input"))
        let input = AudioBuffer(samples: samples, sampleRate: artifact.sampleRate, channels: 1)

        for type in [EmbeddingLoader.EmbeddingType.speech, .music] {
            let provider = USSMLXProvider(
                weightsPath: weights.path,
                embeddingType: type,
                segmentDuration: 2.0,
                useFp16: false
            )
            try await provider.load()
            let output = try await provider.process(input, type: type)
            try expectParity(output.samples, matches: "separated_\(type.rawValue)", in: artifact)
        }
    }
}
