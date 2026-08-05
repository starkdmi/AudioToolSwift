//
//  USSTestWeights.swift
//  AudioToolUSSTests
//
//  Locating USS weights for local test runs
//

import Foundation
import AudioToolUSS
import USSMLXSwift

/// Resolves ResUNet30 weights for tests without requiring a published HuggingFace repo.
///
/// USS weights are no longer bundled - they are ~106 MB and now live on HuggingFace like
/// every other model's. That is right for distribution but would otherwise make a
/// private weights repo block local development, so tests take a path instead.
///
/// Set `AUDIOTOOL_USS_WEIGHTS` to a `resunet30_*.safetensors` file:
///
/// ```bash
/// AUDIOTOOL_USS_WEIGHTS=/path/to/resunet30_fp16.safetensors swift test
/// ```
///
/// Tests that need weights skip themselves when it is unset, so the suite stays green
/// on a machine that has none.
enum USSTestWeights {

    /// Explicit path from the environment, if the file is actually there.
    static var path: String? {
        guard let value = ProcessInfo.processInfo.environment["AUDIOTOOL_USS_WEIGHTS"],
              !value.isEmpty,
              FileManager.default.fileExists(atPath: value)
        else { return nil }
        return value
    }

    /// Whether `path` points at FP16 weights, inferred from the filename.
    static var isFp16: Bool {
        path.map { $0.contains("fp16") } ?? true
    }

    /// A provider backed by local weights, or nil when none are configured.
    static func provider(
        type: EmbeddingLoader.EmbeddingType = .speech,
        segmentDuration: Float = 2.0
    ) -> USSMLXProvider? {
        guard let path else { return nil }
        return USSProviders.separation(
            weightsPath: path,
            type: type,
            segmentDuration: segmentDuration,
            useFp16: isFp16
        )
    }

    /// Message explaining a skip, for test output.
    static let skipReason =
        "set AUDIOTOOL_USS_WEIGHTS to a resunet30 .safetensors file to run USS model tests"
}
