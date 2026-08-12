//
//  PinnedVADModel.swift
//  AudioToolFluidAudio
//
//  Loading Silero VAD from this package's pinned, verified snapshot
//

import CoreML
import Foundation
import AudioToolCore
import FluidAudio
import os

/// Loads the Silero VAD Core ML model from the snapshot ``ModelPins`` fixes.
///
/// `VadManager(config:)` fetches its own weights through FluidAudio's downloader,
/// which resolves `FluidInference/silero-vad-coreml` at its default branch. Pinning
/// the FluidAudio *package* does not pin those bytes: upstream can move the branch
/// and every existing install follows it. The catalog has listed this repository all
/// along, but only for size display and pre-download - nothing made the runtime use
/// what it fetched.
///
/// So the download goes through ``ModelDownloader`` - pinned revision, hash-verified,
/// same as every other model here - and the compiled model is handed to
/// `VadManager(config:vadModel:)`, which exists for exactly this.
///
/// Falling back to FluidAudio's own path is deliberate. This is a 1.1 MB model that
/// several pipelines depend on; if the pinned fetch fails - offline, a repository
/// layout change - degrading to the unpinned-but-working path is better than failing
/// to detect speech at all. The fallback is logged, so "am I pinned?" has an answer.
enum PinnedVADModel {

    private static let logger = Logger(subsystem: "AudioToolSwift", category: "PinnedVAD")

    /// The compiled model directory inside the repository.
    private static let compiledModelName = "silero-vad-unified-256ms-v6.2.1.mlmodelc"

    /// What to fetch: the whole compiled model, whatever it contains.
    private static let downloadFiles = ["\(compiledModelName)/**"]

    /// What must be present before a cached copy counts as installed.
    ///
    /// Not the wildcard used for downloading. `hasRequiredFiles` treats a pattern as
    /// satisfied by any one match, so `…mlmodelc/**` called an interrupted bundle
    /// holding a single file "cached" - the downloader was then never asked to finish
    /// it, loading failed, and the unpinned fallback took over. A compiled Core ML
    /// model is a directory of parts that are individually useless.
    private static let requiredFiles = [
        "\(compiledModelName)/coremldata.bin",
        "\(compiledModelName)/model.mil",
        "\(compiledModelName)/weights/weight.bin",
    ]

    /// A `VadManager` on the pinned model, or on FluidAudio's own copy if the pinned
    /// one cannot be had.
    ///
    /// - Throws: ``AudioToolError/modelIntegrityFailed(repo:details:)`` if the pinned
    ///   snapshot's bytes disagree with the pin. That is the one failure this must
    ///   not paper over: falling back on it would answer "these bytes are not what
    ///   was pinned" by fetching whatever the mutable branch is serving today, which
    ///   inverts the entire point of pinning.
    static func makeManager(config: VadConfig) async throws -> VadManager {
        guard let model = try await loadPinnedModel(computeUnits: config.computeUnits) else {
            return try await VadManager(config: config)
        }
        return VadManager(config: config, vadModel: model)
    }

    /// The pinned model, or `nil` when it could not be fetched or opened for a reason
    /// that says nothing about the bytes - offline, a moved repository, a Core ML
    /// version that will not open it.
    private static func loadPinnedModel(computeUnits: MLComputeUnits) async throws -> MLModel? {
        let directory: URL
        do {
            if let cached = ModelDownloader.shared.localPath(
                for: ModelRepository.sileroVADCoreML,
                matching: requiredFiles
            ) {
                directory = cached
            } else {
                directory = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: ModelRepository.sileroVADCoreML,
                    matching: downloadFiles
                )
            }
        } catch let error as AudioToolError {
            if case .modelIntegrityFailed = error { throw error }
            logFallback(error)
            return nil
        } catch {
            logFallback(error)
            return nil
        }

        let compiled = directory.appendingPathComponent(compiledModelName)
        guard FileManager.default.fileExists(atPath: compiled.path) else {
            logger.warning("Pinned VAD snapshot has no \(compiledModelName, privacy: .public)")
            return nil
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        do {
            return try await MLModel.load(contentsOf: compiled, configuration: configuration)
        } catch {
            logFallback(error)
            return nil
        }
    }

    private static func logFallback(_ error: Error) {
        logger.warning(
            """
            Falling back to FluidAudio's own VAD download; the pinned snapshot \
            could not be loaded: \(error.localizedDescription, privacy: .public)
            """
        )
    }
}
