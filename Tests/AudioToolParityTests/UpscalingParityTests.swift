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
    /// The resampler is deliberately *not* in this number. SR's first step is a
    /// 16 -> 48 kHz upsample, Swift's `AVAudioConverter` and librosa's `soxr_hq`
    /// agree at only about 45 dB, and SR amplifies that until the case reads 22 dB
    /// while everything else sits at round-off. So the reference consumes Swift's
    /// own 48 kHz signal (`Parity/inputs`, written by ``ParityInputExportTests``)
    /// and both sides start from identical samples. What this compares is the
    /// model.
    ///
    /// The provider still resamples internally here, from the 16 kHz `input` - it
    /// has no pre-upsampled entry point - so this passing depends on Swift
    /// reproducing the exported signal, which is deterministic. The resampler
    /// itself is measured separately, in
    /// ``testSuperResolutionResamplerMatchesReference``, against librosa rather
    /// than against Swift's own output.
    func testSuperResolutionMatchesReference() async throws {
        try await assertUpscaling(case: "mossformer2_sr_48k")
    }

    /// A 3-second clip, under the 4-second `maxDirectDuration`, so neither side
    /// chunks. A gap between this and the chunked case is the cost of the seams.
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
    /// match - and it was wrong in the *direct* case too, where nothing chunks. So
    /// the fault was upstream of the model.
    ///
    /// The reference here is `upsampled_48k_librosa`, not `upsampled_48k`. The
    /// latter is the signal Swift exported for the reference to consume, so
    /// measuring Swift against it compares the resampler with itself and reports
    /// `inf` - which is exactly what this test did until 2026-08-10, while its
    /// comment claimed librosa. Since the SR cases hand the reference Swift's own
    /// upsample, this is the only place the resampler is checked against an
    /// implementation that is not the one under test.
    ///
    /// 45 dB is not round-off, and it is not supposed to be: two different
    /// resampler designs agree to about that. The floor exists to catch the
    /// resampler *breaking* - the tail loss and the int64 crossfade both read far
    /// below it - not to claim the two implementations are the same filter.
    func testSuperResolutionResamplerMatchesReference() async throws {
        let artifact = try artifact("mossformer2_sr_48k_direct")
        let input = try XCTUnwrap(artifact.tensor("input"))
        let reference = try XCTUnwrap(artifact.tensor("upsampled_48k_librosa"))

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

        // The assertion, not the diagnostics above: Swift's resampler against
        // librosa's, held to a recorded floor. Without this the seam the SR cases
        // deliberately removed from their own comparison is covered by nothing.
        try expectParity(inMemory, matches: "upsampled_48k_librosa", in: artifact)
    }

    /// Streaming must produce exactly the samples batch produces.
    ///
    /// The provider states that contract and nothing checked it. It is worth
    /// checking now because the substitution spans chunk boundaries: a window
    /// cannot be filtered until its successor's raw output exists, so both paths
    /// carry a one-chunk lookahead, and they have to agree about where the first
    /// chunk's kept region starts and where the last one's real samples end.
    ///
    /// The precedent is specific. These two last diverged when streaming
    /// multiplied the opening chunk by a rising Hann window without normalising
    /// by the accumulated weight, and every stream began with a fade-in from
    /// silence - which took attaching a progress handler to notice, because
    /// that is what switches the pipeline to this path.
    ///
    /// Equality, not a threshold: both paths run the same arithmetic in the same
    /// order on the same device, so any difference at all is a structural bug
    /// rather than round-off.
    func testSuperResolutionStreamMatchesBatch() async throws {
        let artifact = try artifact("mossformer2_sr_48k")
        let weights = try stagedWeights("MossFormer2_SR_48K_MLX", "model_fp32.safetensors")
        let config = try stagedWeights("MossFormer2_SR_48K_MLX", "config.json")
        let samples = try XCTUnwrap(artifact.tensor("input"))
        let input = AudioBuffer(samples: samples, sampleRate: 16_000, channels: 1)

        let provider = MossFormer2SR48KProvider(weightsPath: weights.path, configPath: config.path)
        try await provider.load()

        let batch = try await provider.process(input)

        var streamed: [Float] = []
        for try await piece in provider.processStream(input) {
            streamed.append(contentsOf: piece.samples)
        }

        XCTAssertEqual(
            streamed.count, batch.samples.count,
            "streaming and batch must agree on length"
        )
        let overlap = min(streamed.count, batch.samples.count)
        var worst: (index: Int, diff: Float) = (0, 0)
        for i in 0..<overlap {
            let diff = abs(streamed[i] - batch.samples[i])
            if diff > worst.diff { worst = (i, diff) }
        }
        XCTAssertEqual(
            worst.diff, 0,
            "streaming diverges from batch by \(worst.diff) at sample \(worst.index)"
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
