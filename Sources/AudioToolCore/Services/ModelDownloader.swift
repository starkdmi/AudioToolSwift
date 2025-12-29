//
//  ModelDownloader.swift
//  ClearVoiceCore
//
//  HuggingFace Hub model downloading with progress and cancellation
//

import Foundation
import Hub

/// Progress information for model downloads
public struct DownloadProgress: Sendable {
    /// Fraction completed (0.0 - 1.0)
    public let fractionCompleted: Double
    
    /// Downloaded bytes
    public let completedBytes: Int64
    
    /// Total bytes to download
    public let totalBytes: Int64
    
    /// Download speed in bytes per second (nil if unknown)
    public let bytesPerSecond: Double?
    
    /// Progress percentage (0-100)
    public var percentComplete: Int {
        Int(fractionCompleted * 100)
    }
    
    /// Human-readable speed string
    public var speedString: String? {
        guard let speed = bytesPerSecond else { return nil }
        if speed > 1_000_000 {
            return String(format: "%.1f MB/s", speed / 1_000_000)
        } else if speed > 1_000 {
            return String(format: "%.1f KB/s", speed / 1_000)
        } else {
            return String(format: "%.0f B/s", speed)
        }
    }
}

/// Model download manager using HuggingFace Hub
///
/// Downloads model files from HuggingFace with progress reporting and cancellation.
/// Uses standard HF cache location (`~/.cache/huggingface/hub/`).
///
/// Usage:
/// ```swift
/// // Download with progress
/// for try await progress in ModelDownloader.shared.download(repo: "starkdmi/MossFormer2_SE_48K_MLX") {
///     print("\(progress.percentComplete)% - \(progress.speedString ?? "")")
/// }
///
/// // Check if cached
/// if let path = await ModelDownloader.shared.localPath(for: "starkdmi/MossFormer2_SE_48K_MLX") {
///     print("Model at: \(path)")
/// }
/// ```
public actor ModelDownloader {
    /// Shared instance
    public static let shared = ModelDownloader()
    
    private init() {}
    
    // MARK: - Download
    
    /// Download model files from HuggingFace Hub
    /// - Parameters:
    ///   - repo: Repository ID (e.g., "starkdmi/MossFormer2_SE_48K_MLX")
    ///   - matching: Glob patterns for files to download
    /// - Returns: AsyncThrowingStream of progress updates, final URL on completion
    public func download(
        repo: String,
        matching globs: [String] = ["*.safetensors", "config.json"]
    ) -> AsyncThrowingStream<DownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let hubRepo = Hub.Repo(id: repo)
                    
                    let modelDir = try await HubApi.shared.snapshot(
                        from: hubRepo,
                        matching: globs
                    ) { progress, speed in
                        let downloadProgress = DownloadProgress(
                            fractionCompleted: progress.fractionCompleted,
                            completedBytes: progress.completedUnitCount,
                            totalBytes: progress.totalUnitCount,
                            bytesPerSecond: speed
                        )
                        continuation.yield(downloadProgress)
                    }
                    
                    // Yield final 100% progress
                    continuation.yield(DownloadProgress(
                        fractionCompleted: 1.0,
                        completedBytes: 0,
                        totalBytes: 0,
                        bytesPerSecond: nil
                    ))
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Download model and return final path
    /// - Parameters:
    ///   - repo: Repository ID
    ///   - matching: Glob patterns
    ///   - progress: Progress callback
    /// - Returns: Local directory URL
    public func downloadAndGetPath(
        repo: String,
        matching globs: [String] = ["*.safetensors", "config.json"],
        progress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> URL {
        let hubRepo = Hub.Repo(id: repo)
        
        return try await HubApi.shared.snapshot(
            from: hubRepo,
            matching: globs
        ) { foundationProgress, speed in
            let downloadProgress = DownloadProgress(
                fractionCompleted: foundationProgress.fractionCompleted,
                completedBytes: foundationProgress.completedUnitCount,
                totalBytes: foundationProgress.totalUnitCount,
                bytesPerSecond: speed
            )
            progress(downloadProgress)
        }
    }
    
    // MARK: - Cache Management
    
    /// Check if model is cached locally
    public nonisolated func isDownloaded(repo: String) -> Bool {
        localPath(for: repo) != nil
    }
    
    /// Get local cache path for a repo (nil if not downloaded)
    public nonisolated func localPath(for repo: String) -> URL? {
        let repoName = repo.replacingOccurrences(of: "/", with: "--")
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--\(repoName)")
        
        guard FileManager.default.fileExists(atPath: hfCache.path) else {
            return nil
        }
        
        // Find snapshot directory (contains actual files)
        let snapshotsDir = hfCache.appendingPathComponent("snapshots")
        guard let snapshots = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let firstSnapshot = snapshots.first(where: { !$0.hasPrefix(".") }) else {
            return nil
        }
        
        return snapshotsDir.appendingPathComponent(firstSnapshot)
    }
}
