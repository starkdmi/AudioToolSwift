//
//  ChatterboxParityTests.swift
//  AudioToolParityTests
//
//  Chatterbox conditioning - the half of the model that does not sample.
//

import AudioToolCore
@testable import AudioToolTTS
import AudioToolTestSupport
import MLX
import XCTest

final class ChatterboxParityTests: ParityTestCase {

    /// The shipped 24 kHz path: a 3 s prompt already at S3Gen's rate.
    ///
    /// Nothing resamples on the 24 kHz leg and neither conditioning window bites, so
    /// this case is the straight line through `prepare_conditionals` and not much
    /// else. ``testConditioningMatchesReferenceLong()`` is the one that bends it.
    func testConditioningMatchesReference() async throws {
        try await expectConditioningParity("chatterbox_conditionals_24k")
    }

    /// A 12 s prompt at 22050 Hz, which is awkward on every axis the short case
    /// leaves flat.
    ///
    /// Past both the 6 s encoder and 10 s decoder windows, so the truncations run and
    /// the two token tensors stop being the same tensor - 150 tokens against 250. And
    /// 22050 shares only gcd 150 with 24000, so the polyphase resampler runs at
    /// 160/147 where the short case gets 1/1. A filter with 147 phases is a different
    /// piece of arithmetic from one with none, and this is the case that runs it.
    func testConditioningMatchesReferenceLong() async throws {
        try await expectConditioningParity("chatterbox_conditionals_22k_long")
    }

    /// `prepare_conditionals`, both sides.
    ///
    /// The generated waveform is not comparable: `t3.inference` samples with a
    /// temperature, and MLX's RNG stream is not shared between the Python and Swift
    /// runtimes, so equal seeds do not buy equal draws. Conditioning is what remains
    /// deterministic, and it is not a small remainder - the voice encoder, the S3
    /// tokenizer, the log-mel front end and CAMPPlus all run here, and every one of
    /// them feeds both branches downstream. If these agree, a difference in the audio
    /// is the sampler. If they disagree, the audio never had a chance.
    ///
    /// Read a failure here as a question about both sides rather than a verdict on
    /// the Swift. This case has already moved the reference once: `model.py`
    /// resampled with librosa, which is soxr and LGPL, and both sides were moved onto
    /// one published design so the Swift could reproduce it at all. The reasoning is
    /// in `ParityThresholds`.
    /// Each published checkpoint loads without disturbing the conditioning path.
    ///
    /// **This is not quantization parity, and must not be read as it.** Counted from
    /// `chatterbox-4bit/model.safetensors`: `ve` has 13 keys and none of them
    /// quantized, `s3gen.speaker_encoder` has 815 and none quantized. Every module
    /// feeding the five tensors below is stored at full precision in the quantized
    /// checkpoints, so all five come out bit-identical to the fp32 case - measured,
    /// not assumed: 116.7 dB and maxabs 3.465e-07 on `ve_speaker_emb` at fp32, 4bit,
    /// 6bit and 8bit alike.
    ///
    /// Quantization lives in `t3` (219 modules) and `s3gen.flow` (423) - the
    /// autoregressive token model and the flow decoder, which are exactly the
    /// sampled halves that `expectConditioningParity` exists to exclude. Verifying
    /// those needs a deterministic sub-run of `t3` rather than this.
    ///
    /// What these four do earn: a quantized checkpoint loads, its unquantized
    /// modules survive the load intact, and Swift and Python agree on which modules
    /// to leave alone. A load path that mangled `ve` while quantizing `t3` would
    /// fail here. That is worth a test - it is just a much narrower claim than the
    /// name "parity at 4bit" suggests.
    ///
    /// FP16 is the exception and is a real comparison: it is a dtype change applied
    /// to every weight, `ve` included, which is why it measures 118.6 dB where the
    /// integer cases repeat fp32's 116.7 exactly.
    func testConditioningMatchesReferenceFP16() async throws {
        try await expectConditioningParity("chatterbox_conditionals_24k_fp16", repo: "chatterbox-fp16")
    }

    func testConditioningMatchesReference8Bit() async throws {
        try await expectConditioningParity("chatterbox_conditionals_24k_8bit", repo: "chatterbox-8bit")
    }

    func testConditioningMatchesReference6Bit() async throws {
        try await expectConditioningParity("chatterbox_conditionals_24k_6bit", repo: "chatterbox-6bit")
    }

    func testConditioningMatchesReference4Bit() async throws {
        try await expectConditioningParity("chatterbox_conditionals_24k_4bit", repo: "chatterbox-4bit")
    }

    private func expectConditioningParity(
        _ caseName: String, repo: String = "chatterbox", line: UInt = #line
    ) async throws {
        let artifact = try artifact(caseName)
        let weights = try stagedWeights(repo, "model.safetensors")

        let samples = try XCTUnwrap(artifact.tensor("input"))

        let provider = ChatterboxTTSProvider(modelPath: weights.deletingLastPathComponent())
        try await provider.load()
        // The artifact's rate is the rate of `input`, not the model's 24 kHz output:
        // for the long case those differ, and passing the model rate here would
        // resample the prompt before conditioning ever started.
        try await provider.setReferenceAudio(
            AudioBuffer(samples: samples, sampleRate: artifact.sampleRate, channels: 1)
        )

        let conditioning = await provider.conditioningTensors()

        // Named individually rather than looped, so a failure says which stage moved
        // instead of "conditioning differs".
        try expect(conditioning, "t3.speaker_emb", matches: "ve_speaker_emb", in: artifact)
        try expect(
            conditioning, "t3.cond_prompt_speech_tokens",
            matches: "t3_cond_prompt_tokens", in: artifact
        )
        try expect(conditioning, "gen.prompt_token", matches: "s3gen_prompt_token", in: artifact)
        try expect(conditioning, "gen.prompt_feat", matches: "s3gen_prompt_feat", in: artifact)
        try expect(conditioning, "gen.embedding", matches: "s3gen_embedding", in: artifact)
    }

    private func expect(
        _ tensors: [String: ChatterboxTTSProvider.ConditioningSnapshot],
        _ key: String,
        matches artifactTensor: String,
        in artifact: ParityArtifact,
        line: UInt = #line
    ) throws {
        guard let snapshot = tensors[key] else {
            XCTFail(
                "provider produced no \(key); it has \(tensors.keys.sorted())",
                line: line
            )
            return
        }
        if let expected = artifact.shape(artifactTensor), expected != snapshot.shape {
            XCTFail(
                "\(key) is \(snapshot.shape), reference \(artifactTensor) is \(expected) - "
                + "shape first, the sample comparison below is only meaningful if these agree",
                line: line
            )
        }
        try expectParity(snapshot.values, matches: artifactTensor, in: artifact, line: line)
    }
}
