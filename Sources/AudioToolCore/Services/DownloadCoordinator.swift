//
//  DownloadCoordinator.swift
//  ClearVoiceCore
//
//  Central coordinator for all model downloads with cancellation support
//

import Foundation
import Hub

/// Central coordinator for all model downloads
///
/// Thread-safe actor that manages concurrent downloads, cancellation, and storage checks.
///
/// Usage:
/// ```swift
/// // Start a download
/// for try await progress in await DownloadCoordinator.shared.download(variant: variant) {
///     print("\(progress.percentComplete)%")
/// }
///
/// // Cancel a download
/// await DownloadCoordinator.shared.cancel(variantId: "mossformer2_se_int8")
///
/// // Delete an installed model
/// try await DownloadCoordinator.shared.delete(variant: variant)
/// ```
public actor DownloadCoordinator {
    /// Shared instance
    public static let shared = DownloadCoordinator()
    
    /// Active download tasks by variant ID
    private var activeTasks: [String: Task<Void, Never>] = [:]
    
    /// Maximum concurrent downloads
    private let maxConcurrentDownloads = 2
    
    /// Storage buffer to require beyond model size (50 MB)
    private let storageBuffer: Int64 = 50 * 1024 * 1024
    
    private init() {}
    
    // MARK: - Download
    
    /// Download a model variant with progress reporting
    /// - Parameters:
    ///   - variant: The model variant to download
    ///   - priority: Task priority for the download
    /// - Returns: AsyncThrowingStream of download progress
    public func download(
        variant: ModelVariant,
        priority: TaskPriority = .medium
    ) -> AsyncThrowingStream<DownloadProgress, Error> {
        // Check if already downloading synchronously before creating stream
        if activeTasks[variant.id] != nil {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: DownloadError.unknownError(
                    underlying: NSError(domain: "DownloadCoordinator", code: 1,
                                        userInfo: [NSLocalizedDescriptionKey: "Already downloading"])
                ))
            }
        }
        
        // Check concurrent download limit
        if activeTasks.count >= maxConcurrentDownloads {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: DownloadError.unknownError(
                    underlying: NSError(domain: "DownloadCoordinator", code: 2,
                                        userInfo: [NSLocalizedDescriptionKey: "Maximum concurrent downloads reached (\(maxConcurrentDownloads))"])
                ))
            }
        }
        
        // Create placeholder task to reserve the slot synchronously
        let placeholderTask = Task<Void, Never> { }
        activeTasks[variant.id] = placeholderTask
        
        return AsyncThrowingStream { continuation in
            let task = Task(priority: priority) {
                do {
                    // Check storage availability
                    try await self.checkStorageAvailable(for: variant)
                    
                    // Check for cancellation
                    try Task.checkCancellation()
                    
                    // Start HuggingFace Hub download
                    let hubRepo = Hub.Repo(id: variant.repo)
                    
                    let modelDir = try await HubApi.shared.snapshot(
                        from: hubRepo,
                        matching: variant.files
                    ) { progress, speed in
                        // Check for cancellation
                        guard !Task.isCancelled else { return }
                        
                        let downloadProgress = DownloadProgress(
                            fractionCompleted: progress.fractionCompleted,
                            completedBytes: progress.completedUnitCount,
                            totalBytes: progress.totalUnitCount,
                            bytesPerSecond: speed
                        )
                        continuation.yield(downloadProgress)
                    }
                    
                    // Check for cancellation after download
                    try Task.checkCancellation()
                    
                    // Verify download
                    try await self.verifyDownload(at: modelDir, for: variant)
                    
                    // Yield final 100% progress
                    continuation.yield(DownloadProgress(
                        fractionCompleted: 1.0,
                        completedBytes: variant.sizeBytes,
                        totalBytes: variant.sizeBytes,
                        bytesPerSecond: nil
                    ))
                    
                    continuation.finish()
                    
                } catch is CancellationError {
                    continuation.finish(throwing: DownloadError.cancelled)
                } catch {
                    continuation.finish(throwing: error)
                }
                
                // Clean up
                self.removeTask(for: variant.id)
            }
            
            // Replace placeholder with real task
            Task {
                self.registerTask(task, for: variant.id)
            }
            
            // Handle stream termination
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
    
    /// Download a model variant and return the local path
    /// - Parameters:
    ///   - variant: The model variant to download
    ///   - progress: Progress callback
    /// - Returns: Local directory URL containing model files
    public func downloadAndGetPath(
        variant: ModelVariant,
        progress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> URL {
        // Check if already downloading
        if activeTasks[variant.id] != nil {
            throw DownloadError.unknownError(
                underlying: NSError(domain: "DownloadCoordinator", code: 1,
                                    userInfo: [NSLocalizedDescriptionKey: "Already downloading"])
            )
        }
        
        // Check concurrent download limit
        if activeTasks.count >= maxConcurrentDownloads {
            throw DownloadError.unknownError(
                underlying: NSError(domain: "DownloadCoordinator", code: 2,
                                    userInfo: [NSLocalizedDescriptionKey: "Maximum concurrent downloads reached (\(maxConcurrentDownloads))"])
            )
        }
        
        // Check storage
        try await checkStorageAvailable(for: variant)
        
        // Register placeholder to reserve slot
        let placeholderTask = Task<Void, Never> { }
        activeTasks[variant.id] = placeholderTask
        
        defer {
            activeTasks.removeValue(forKey: variant.id)
        }
        
        let hubRepo = Hub.Repo(id: variant.repo)
        
        return try await HubApi.shared.snapshot(
            from: hubRepo,
            matching: variant.files
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
    
    // MARK: - Cancellation
    
    /// Cancel a specific download
    /// - Parameter variantId: ID of the variant to cancel
    public func cancel(variantId: String) {
        activeTasks[variantId]?.cancel()
        activeTasks.removeValue(forKey: variantId)
    }
    
    /// Cancel all active downloads
    public func cancelAll() {
        for task in activeTasks.values {
            task.cancel()
        }
        activeTasks.removeAll()
    }
    
    /// Check if a variant is currently downloading
    /// - Parameter variantId: Variant ID to check
    /// - Returns: True if download is in progress
    public func isDownloading(variantId: String) -> Bool {
        activeTasks[variantId] != nil
    }
    
    /// Get list of currently downloading variant IDs
    public var activeDownloadIds: [String] {
        Array(activeTasks.keys)
    }
    
    // MARK: - Deletion
    
    /// Delete an installed model
    /// - Parameter variant: The model variant to delete
    public func delete(variant: ModelVariant) async throws {
        // Cancel if currently downloading
        cancel(variantId: variant.id)
        
        // Get local path
        guard let localPath = ModelDownloader.shared.localPath(for: variant.repo) else {
            return // Already deleted or never downloaded
        }
        
        // Remove the snapshot directory
        try FileManager.default.removeItem(at: localPath)
        
        // Try to clean up empty parent directories in HF cache
        cleanupEmptyHFCache(for: variant.repo)
    }
    
    /// Delete all installed models
    public func deleteAll(variants: [ModelVariant]) async {
        cancelAll()
        
        for variant in variants {
            try? await delete(variant: variant)
        }
    }
    
    // MARK: - Package Downloads
    
    /// Download all variants in a package
    /// - Parameters:
    ///   - package: The package to download
    ///   - registry: Model registry to look up variants
    /// - Returns: AsyncThrowingStream of package download progress
    public func downloadPackage(
        _ package: ModelPackage,
        registry: ModelRegistry
    ) -> AsyncThrowingStream<PackageDownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var completed = 0
                let total = package.variantIds.count
                
                for variantId in package.variantIds {
                    // Check for cancellation
                    guard !Task.isCancelled else {
                        continuation.finish(throwing: DownloadError.cancelled)
                        return
                    }
                    
                    guard let variant = registry.variant(id: variantId) else {
                        continuation.finish(throwing: DownloadError.variantNotFound(id: variantId))
                        return
                    }
                    
                    // Skip if already downloaded
                    if ModelDownloader.shared.isDownloaded(repo: variant.repo) {
                        completed += 1
                        continuation.yield(PackageDownloadProgress(
                            currentVariant: variant,
                            variantProgress: DownloadProgress(
                                fractionCompleted: 1.0,
                                completedBytes: variant.sizeBytes,
                                totalBytes: variant.sizeBytes,
                                bytesPerSecond: nil
                            ),
                            completedCount: completed,
                            totalCount: total
                        ))
                        continue
                    }
                    
                    // Download this variant
                    do {
                        for try await progress in download(variant: variant) {
                            continuation.yield(PackageDownloadProgress(
                                currentVariant: variant,
                                variantProgress: progress,
                                completedCount: completed,
                                totalCount: total
                            ))
                        }
                        completed += 1
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                
                continuation.finish()
            }
        }
    }
    
    // MARK: - Storage
    
    /// Check available storage for a variant
    /// - Parameter variant: Variant to check storage for
    /// - Throws: DownloadError.insufficientStorage if not enough space
    public func checkStorageAvailable(for variant: ModelVariant) async throws {
        let availableBytes = try FileManager.default.availableStorageSpace()
        let requiredBytes = variant.sizeBytes + storageBuffer
        
        guard availableBytes > requiredBytes else {
            throw DownloadError.insufficientStorage(
                required: requiredBytes,
                available: availableBytes
            )
        }
    }
    
    /// Check available storage for a package
    /// - Parameters:
    ///   - package: Package to check storage for
    ///   - registry: Model registry to look up variants
    /// - Throws: DownloadError.insufficientStorage if not enough space
    public func checkStorageAvailable(
        for package: ModelPackage,
        registry: ModelRegistry
    ) async throws {
        let availableBytes = try FileManager.default.availableStorageSpace()
        
        // Calculate needed space (excluding already downloaded)
        var requiredBytes: Int64 = storageBuffer
        for variantId in package.variantIds {
            guard let variant = registry.variant(id: variantId),
                  !ModelDownloader.shared.isDownloaded(repo: variant.repo) else {
                continue
            }
            requiredBytes += variant.sizeBytes
        }
        
        guard availableBytes > requiredBytes else {
            throw DownloadError.insufficientStorage(
                required: requiredBytes,
                available: availableBytes
            )
        }
    }
    
    // MARK: - Private Helpers
    
    private func registerTask(_ task: Task<Void, Never>, for variantId: String) {
        activeTasks[variantId] = task
    }
    
    private func removeTask(for variantId: String) {
        activeTasks.removeValue(forKey: variantId)
    }
    
    private func verifyDownload(at url: URL, for variant: ModelVariant) async throws {
        // Verify expected files exist (skip glob patterns)
        for pattern in variant.files {
            if pattern.contains("*") { continue }
            let fileURL = url.appendingPathComponent(pattern)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                throw DownloadError.fileVerificationFailed
            }
        }
    }
    
    private nonisolated func cleanupEmptyHFCache(for repo: String) {
        let repoName = repo.replacingOccurrences(of: "/", with: "--")
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--\(repoName)")
        
        // Check if snapshots directory is empty or only has hidden files
        let snapshotsDir = hfCache.appendingPathComponent("snapshots")
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path) {
            let visibleContents = contents.filter { !$0.hasPrefix(".") }
            if visibleContents.isEmpty {
                // Remove the entire model cache directory
                try? FileManager.default.removeItem(at: hfCache)
            }
        }
    }
}
