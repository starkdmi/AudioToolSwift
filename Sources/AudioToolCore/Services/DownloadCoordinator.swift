//
//  DownloadCoordinator.swift
//  AudioToolCore
//
//  Central coordinator for all model downloads with cancellation support
//

import Foundation

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
    
    private struct ActiveDownload: Sendable {
        let token: UUID
        let cancel: @Sendable () -> Void
        let waitForCompletion: @Sendable () async -> Void
    }

    /// Active real download operations by variant ID. Tokens make cleanup safe
    /// when a cancelled download is immediately restarted.
    private var activeTasks: [String: ActiveDownload] = [:]
    
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
        
        let token = UUID()
        let pair = AsyncThrowingStream<DownloadProgress, Error>.makeStream()
        let task = Task(priority: priority) {
            do {
                try await self.checkStorageAvailable(for: variant)
                try Task.checkCancellation()

                let modelDir = try await ModelDownloader.shared.downloadAndGetPath(
                    repo: variant.repo,
                    matching: variant.files
                ) { progress in
                    _ = pair.continuation.yield(progress)
                }

                try Task.checkCancellation()
                try await self.verifyDownload(at: modelDir, for: variant)
                try Task.checkCancellation()
                _ = pair.continuation.yield(DownloadProgress(
                    fractionCompleted: 1,
                    completedBytes: variant.sizeBytes,
                    totalBytes: variant.sizeBytes,
                    bytesPerSecond: nil
                ))
                pair.continuation.finish()
            } catch is CancellationError {
                pair.continuation.finish(throwing: DownloadError.cancelled)
            } catch {
                pair.continuation.finish(
                    throwing: Task.isCancelled ? DownloadError.cancelled : error
                )
            }
            self.removeTask(for: variant.id, token: token)
        }
        activeTasks[variant.id] = ActiveDownload(
            token: token,
            cancel: { task.cancel() },
            waitForCompletion: { await task.value }
        )
        pair.continuation.onTermination = { @Sendable _ in task.cancel() }
        return pair.stream
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
        
        let token = UUID()
        let task = Task<URL, Error> {
            try await self.checkStorageAvailable(for: variant)
            try Task.checkCancellation()
            let url = try await ModelDownloader.shared.downloadAndGetPath(
                repo: variant.repo,
                matching: variant.files,
                progress: progress
            )
            try Task.checkCancellation()
            try await self.verifyDownload(at: url, for: variant)
            return url
        }
        activeTasks[variant.id] = ActiveDownload(
            token: token,
            cancel: { task.cancel() },
            waitForCompletion: { _ = try? await task.value }
        )

        do {
            let url = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            removeTask(for: variant.id, token: token)
            return url
        } catch is CancellationError {
            removeTask(for: variant.id, token: token)
            throw DownloadError.cancelled
        } catch {
            removeTask(for: variant.id, token: token)
            if task.isCancelled { throw DownloadError.cancelled }
            throw error
        }
    }
    
    // MARK: - Cancellation
    
    /// Cancel a specific download
    /// - Parameter variantId: ID of the variant to cancel
    public func cancel(variantId: String) async {
        guard let operation = activeTasks[variantId] else { return }
        operation.cancel()
        await operation.waitForCompletion()
        removeTask(for: variantId, token: operation.token)
    }
    
    /// Cancel all active downloads
    public func cancelAll() async {
        let operations = activeTasks
        for operation in operations.values {
            operation.cancel()
        }
        await withTaskGroup(of: Void.self) { group in
            for operation in operations.values {
                group.addTask { await operation.waitForCompletion() }
            }
            await group.waitForAll()
        }
        for (variantId, operation) in operations {
            removeTask(for: variantId, token: operation.token)
        }
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
        await cancel(variantId: variant.id)
        
        let siblings = ModelCatalog.shared.allVariants.filter { $0.repo == variant.repo }
        try await ModelDownloader.shared.delete(variant: variant, preserving: siblings)
    }
    
    /// Delete all installed models
    public func deleteAll(variants: [ModelVariant]) async {
        await cancelAll()
        
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
        registry: ModelCatalog
    ) -> AsyncThrowingStream<PackageDownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
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

                    // Join an operation that won the ownership race. A package
                    // may report the variant complete only after that operation
                    // finishes and the exact variant files verify successfully.
                    if let operation = activeTasks[variant.id] {
                        do {
                            await operation.waitForCompletion()
                            try Task.checkCancellation()
                            guard let modelDirectory = ModelDownloader.shared.localPath(
                                for: variant
                            ) else {
                                throw DownloadError.fileVerificationFailed
                            }
                            try await verifyDownload(at: modelDirectory, for: variant)
                        } catch is CancellationError {
                            continuation.finish(throwing: DownloadError.cancelled)
                            return
                        } catch {
                            continuation.finish(throwing: error)
                            return
                        }

                        completed += 1
                        if case .terminated = continuation.yield(PackageDownloadProgress(
                            currentVariant: variant,
                            variantProgress: DownloadProgress(
                                fractionCompleted: 1.0,
                                completedBytes: variant.sizeBytes,
                                totalBytes: variant.sizeBytes,
                                bytesPerSecond: nil
                            ),
                            completedCount: completed,
                            totalCount: total,
                            isCurrentVariantVerified: true
                        )) { return }
                        continue
                    }
                    
                    // Skip if already downloaded
                    if ModelDownloader.shared.isDownloaded(variant: variant) {
                        completed += 1
                        if case .terminated = continuation.yield(PackageDownloadProgress(
                            currentVariant: variant,
                            variantProgress: DownloadProgress(
                                fractionCompleted: 1.0,
                                completedBytes: variant.sizeBytes,
                                totalBytes: variant.sizeBytes,
                                bytesPerSecond: nil
                            ),
                            completedCount: completed,
                            totalCount: total,
                            isCurrentVariantVerified: true
                        )) { return }
                        continue
                    }
                    
                    // Download this variant
                    do {
                        for try await progress in download(variant: variant) {
                            if case .terminated = continuation.yield(PackageDownloadProgress(
                                currentVariant: variant,
                                variantProgress: progress,
                                completedCount: completed,
                                totalCount: total
                            )) { return }
                        }
                        completed += 1
                        if case .terminated = continuation.yield(PackageDownloadProgress(
                            currentVariant: variant,
                            variantProgress: DownloadProgress(
                                fractionCompleted: 1.0,
                                completedBytes: variant.sizeBytes,
                                totalBytes: variant.sizeBytes,
                                bytesPerSecond: nil
                            ),
                            completedCount: completed,
                            totalCount: total,
                            isCurrentVariantVerified: true
                        )) { return }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in producer.cancel() }
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
        registry: ModelCatalog
    ) async throws {
        let availableBytes = try FileManager.default.availableStorageSpace()
        
        // Calculate needed space (excluding already downloaded)
        var requiredBytes: Int64 = storageBuffer
        for variantId in package.variantIds {
            guard let variant = registry.variant(id: variantId),
                  !ModelDownloader.shared.isDownloaded(variant: variant) else {
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
    
    private func removeTask(for variantId: String, token: UUID) {
        guard activeTasks[variantId]?.token == token else { return }
        activeTasks.removeValue(forKey: variantId)
    }
    
    private func verifyDownload(at url: URL, for variant: ModelVariant) async throws {
        guard ModelDownloader.hasRequiredFiles(at: url, patterns: variant.files) else {
            throw DownloadError.fileVerificationFailed
        }
    }
    
}
