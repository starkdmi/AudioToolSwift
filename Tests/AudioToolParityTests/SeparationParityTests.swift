//
//  SeparationParityTests.swift
//  AudioToolParityTests
//
//  MossFormer2 SS (three configurations) and Demucs.
//

import AudioToolCore
import AudioToolMLX
import AudioToolTestSupport
import XCTest

final class SeparationParityTests: ParityTestCase {

    // MARK: - Speaker separation

    /// Each configuration is its own test, so a failure names the config.
    ///
    /// Compared against `speaker_N_normalized`: the provider RMS-matches to the
    /// input and peak-clamps before returning, exactly as the reference's
    /// `save_sources` does at write time. Raw SS output runs 50-200x the input
    /// level, so this is not cosmetic - see `speaker_N` in the artifact for the
    /// unscaled tensors.
    /// Each config gets two: the chunked path a caller hits above four seconds,
    /// and a short clip where neither side chunks. If the direct one is exact and
    /// the chunked one is not, the seams are where to look - which is a question
    /// no single case can answer.
    func testTwoSpeaker16kMatchesReference() async throws {
        try await assertSeparation(
            case: "mossformer2_ss_2spk_16k", model: .twoSpeaker,
            repo: "MossFormer2_SS_2SPK_16K_MLX", speakers: 2
        )
    }

    func testTwoSpeaker16kDirectMatchesReference() async throws {
        try await assertSeparation(
            case: "mossformer2_ss_2spk_16k_direct", model: .twoSpeaker,
            repo: "MossFormer2_SS_2SPK_16K_MLX", speakers: 2
        )
    }

    func testTwoSpeakerWHAMR8kMatchesReference() async throws {
        try await assertSeparation(
            case: "mossformer2_ss_2spk_whamr_8k", model: .twoSpeakerWHAMR,
            repo: "MossFormer2_SS_2SPK_WHAMR_8K_MLX", speakers: 2
        )
    }

    func testTwoSpeakerWHAMR8kDirectMatchesReference() async throws {
        try await assertSeparation(
            case: "mossformer2_ss_2spk_whamr_8k_direct", model: .twoSpeakerWHAMR,
            repo: "MossFormer2_SS_2SPK_WHAMR_8K_MLX", speakers: 2
        )
    }

    func testThreeSpeaker8kMatchesReference() async throws {
        try await assertSeparation(
            case: "mossformer2_ss_3spk_8k", model: .threeSpeaker,
            repo: "MossFormer2_SS_3SPK_8K_MLX", speakers: 3
        )
    }

    func testThreeSpeaker8kDirectMatchesReference() async throws {
        try await assertSeparation(
            case: "mossformer2_ss_3spk_8k_direct", model: .threeSpeaker,
            repo: "MossFormer2_SS_3SPK_8K_MLX", speakers: 3
        )
    }

    private func assertSeparation(
        case caseName: String,
        model: MossFormer2SSProvider.Model,
        repo: String,
        speakers: Int
    ) async throws {
        let artifact = try artifact(caseName)
        let weights = try stagedWeights(repo, "model_fp32.safetensors")
        let samples = try XCTUnwrap(artifact.tensor("input"))

        let provider = MossFormer2SSProvider(model: model, weightsPath: weights.path)
        try await provider.load()
        let outputs = try await provider.separate(
            AudioBuffer(samples: samples, sampleRate: artifact.sampleRate, channels: 1)
        )

        XCTAssertEqual(outputs.count, speakers, "\(caseName): expected \(speakers) sources")
        // Speaker order is part of what is pinned. A permutation would be invisible
        // to any combined metric and obvious here.
        for (index, output) in outputs.enumerated() {
            try expectParity(
                output.samples, matches: "speaker_\(index + 1)_normalized", in: artifact
            )
        }
    }

    // MARK: - Music separation

    /// Demucs, stereo in.
    ///
    /// Two details this case exists to pin, both of which have bitten:
    ///
    /// 1. `AudioBuffer.samples` is *interleaved*; the artifact stores stems
    ///    *planar*, one tensor per channel. The input is interleaved here on
    ///    purpose - reading interleaved data as planar is precisely the bug that
    ///    was fixed in this provider.
    /// 2. `separate(_:stem:)` returns mono - it means over channels before it
    ///    returns. So the reference for the public API is the mean of the two
    ///    recorded channels, not either one.
    func testDemucsMatchesReference() async throws {
        let artifact = try artifact("demucs_vocals_44k")
        let weights = try reference("Models/demucs_mlx_swift/Weights")

        let planar = try XCTUnwrap(artifact.tensor("input"))
        let shape = try XCTUnwrap(artifact.shape("input"))
        XCTAssertEqual(shape.count, 2, "demucs input should be (channels, frames)")
        XCTAssertEqual(shape[0], 2, "demucs input should be stereo")
        let frames = shape[1]

        var interleaved = [Float](repeating: 0, count: planar.count)
        for frame in 0..<frames {
            interleaved[frame * 2] = planar[frame]
            interleaved[frame * 2 + 1] = planar[frames + frame]
        }

        let provider = DemucsProvider(weightsDirectory: weights.path)
        try await provider.loadAll()

        let input = AudioBuffer(samples: interleaved, sampleRate: artifact.sampleRate, channels: 2)
        for stem in DemucsProvider.Stem.allCases {
            let output = try await provider.separate(input, stem: stem)
            try expectParity(output.samples, matches: "\(stem.rawValue)_mono", in: artifact)
        }
    }
}
