//
//  ModelManager.swift
//  AudioTool
//
//  SwiftUI-ready observable model manager for download UI
//

import Foundation
import AudioToolCore

#if canImport(Observation)
import Observation
#endif

/// Observable model manager for SwiftUI integration
///
/// Provides reactive state for model downloads, installations, and deletions.
///
/// Usage:
/// ```swift
/// struct ContentView: View {
///     @State private var manager = ModelManager.shared
///
///     var body: some View {
///         List(manager.registry.models) { model in
///             ModelDownloadCard(model: model, manager: manager)
///         }
///     }
/// }
/// ```
@available(macOS 14.0, iOS 17.0, *)
@MainActor
@Observable
public final class ModelManager {
    /// Shared instance
    public static let shared = ModelManager()
    
    // MARK: - State
    
    /// Active download tasks by variant ID
    public private(set) var downloadTasks: [String: DownloadTask] = [:]
    
    /// Set of installed variant IDs
    public private(set) var installedVariantIds: Set<String> = []
    
    /// Model registry
    public let registry: ModelCatalog
    
    /// Last error encountered
    public private(set) var lastError: Error?
    
    /// Whether initial scan is complete
    public private(set) var isInitialized: Bool = false
    
    private init() {
        self.registry = ModelCatalog.shared
        Task {
            await refreshInstalledModels()
            isInitialized = true
        }
    }
    
    // MARK: - Download Management
    
    /// Start downloading a variant
    /// - Parameter variant: The variant to download
    public func download(_ variant: ModelVariant) {
        // Skip if already downloading or installed
        guard downloadTasks[variant.id] == nil,
              !installedVariantIds.contains(variant.id) else {
            return
        }
        
        // Create task entry
        downloadTasks[variant.id] = DownloadTask(
            id: variant.id,
            variant: variant,
            status: .downloading,
            progress: nil,
            startedAt: Date()
        )
        
        // Start download
        Task {
            do {
                for try await progress in await DownloadCoordinator.shared.download(variant: variant) {
                    // Update progress on main actor
                    downloadTasks[variant.id]?.progress = progress
                    downloadTasks[variant.id]?.status = .downloading
                }
                
                // Mark as completed
                downloadTasks[variant.id]?.status = .completed
                installedVariantIds.insert(variant.id)
                
                // Clear from active tasks after short delay
                try? await Task.sleep(for: .seconds(1.5))
                downloadTasks.removeValue(forKey: variant.id)
                
            } catch let error as DownloadError {
                downloadTasks[variant.id]?.status = .failed(error.localizedDescription)
                downloadTasks[variant.id]?.errorMessage = error.localizedDescription
                lastError = error
            } catch {
                downloadTasks[variant.id]?.status = .failed(error.localizedDescription)
                downloadTasks[variant.id]?.errorMessage = error.localizedDescription
                lastError = error
            }
        }
    }
    
    /// Cancel a download in progress
    /// - Parameter variantId: ID of the variant to cancel
    public func cancel(_ variantId: String) {
        Task {
            await DownloadCoordinator.shared.cancel(variantId: variantId)
            downloadTasks[variantId]?.status = .cancelled
            
            // Remove after short delay
            try? await Task.sleep(for: .milliseconds(500))
            downloadTasks.removeValue(forKey: variantId)
        }
    }
    
    /// Delete an installed model
    /// - Parameter variant: The variant to delete
    public func delete(_ variant: ModelVariant) {
        Task {
            do {
                try await DownloadCoordinator.shared.delete(variant: variant)
                installedVariantIds.remove(variant.id)
            } catch {
                lastError = error
            }
        }
    }
    
    /// Download an entire package
    /// - Parameter package: The package to download
    public func downloadPackage(_ package: ModelPackage) {
        Task {
            let coordinatorActiveIds = Set(
                await DownloadCoordinator.shared.activeDownloadIds
            )
            let uiActiveIds = Set(downloadTasks.compactMap { variantId, task in
                task.status.isActive ? variantId : nil
            })
            guard let requestedPackage = Self.packageForDownload(
                package,
                excluding: coordinatorActiveIds.union(uiActiveIds),
                registry: registry
            ) else {
                // Every requested variant already has an owner. Leave those
                // operations and their UI state untouched.
                return
            }
            let ownedVariantIds = Set(requestedPackage.variantIds)
            let initialTasks = Self.packageDownloadTasks(
                for: requestedPackage,
                registry: registry,
                startedAt: Date()
            )
            for (variantId, task) in initialTasks {
                // Reset stale failed/completed rows for variants this package now
                // owns. This also guarantees the catch path has a retryable entry
                // when storage or Hub setup fails before the first progress event.
                downloadTasks[variantId] = task
            }

            do {
                let stream = await DownloadCoordinator.shared.downloadPackage(
                    requestedPackage,
                    registry: registry
                )
                for try await packageProgress in stream {
                    let variant = packageProgress.currentVariant
                    if downloadTasks[variant.id] == nil {
                        downloadTasks[variant.id] = DownloadTask(
                            id: variant.id,
                            variant: variant,
                            status: .downloading,
                            progress: nil,
                            startedAt: Date()
                        )
                    }
                    downloadTasks[variant.id]?.progress = packageProgress.variantProgress
                    if packageProgress.isCurrentVariantVerified {
                        downloadTasks[variant.id]?.status = .completed
                        installedVariantIds.insert(variant.id)
                    } else {
                        // Hub byte progress can reach 100% before cancellation
                        // checks and file verification have completed.
                        downloadTasks[variant.id]?.status = .downloading
                    }
                }

                let completedIds = requestedPackage.variantIds.filter {
                    downloadTasks[$0]?.status == .completed
                }
                try? await Task.sleep(for: .seconds(1.5))
                for id in completedIds { downloadTasks.removeValue(forKey: id) }
            } catch {
                lastError = error
                let completedIds = Self.completedTaskIdsToRemoveAfterPackageFailure(
                    ownedVariantIds: ownedVariantIds,
                    downloadTasks: downloadTasks
                )
                for variantId in ownedVariantIds where downloadTasks[variantId]?.status.isActive == true {
                    downloadTasks[variantId]?.status = .failed(error.localizedDescription)
                    downloadTasks[variantId]?.errorMessage = error.localizedDescription
                    installedVariantIds.remove(variantId)
                }
                // Earlier variants remain installed, but their transient success
                // rows must not outlive the failed package operation. A stale row
                // would make download(_:) reject a later reinstall after deletion.
                for variantId in completedIds {
                    downloadTasks.removeValue(forKey: variantId)
                }
            }
        }
    }
    
    /// Cancel all active downloads
    public func cancelAll() {
        Task {
            await DownloadCoordinator.shared.cancelAll()
            for key in downloadTasks.keys {
                downloadTasks[key]?.status = .cancelled
            }
            try? await Task.sleep(for: .milliseconds(500))
            downloadTasks.removeAll()
        }
    }
    
    /// Retry a failed download
    /// - Parameter variantId: ID of the variant to retry
    public func retry(_ variantId: String) {
        guard let task = downloadTasks[variantId],
              case .failed = task.status else {
            return
        }
        
        // Remove failed task
        downloadTasks.removeValue(forKey: variantId)
        
        // Restart download
        download(task.variant)
    }
    
    /// Clear last error
    public func clearError() {
        lastError = nil
    }
    
    // MARK: - State Queries
    
    /// Check if a variant is downloaded
    /// - Parameter variantId: Variant ID to check
    /// - Returns: True if installed
    public func isDownloaded(_ variantId: String) -> Bool {
        installedVariantIds.contains(variantId)
    }
    
    /// Check if a variant is currently downloading
    /// - Parameter variantId: Variant ID to check
    /// - Returns: True if downloading
    public func isDownloading(_ variantId: String) -> Bool {
        guard let task = downloadTasks[variantId] else { return false }
        return task.status.isActive
    }
    
    /// Get download progress for a variant
    /// - Parameter variantId: Variant ID
    /// - Returns: DownloadProgress or nil
    public func progress(for variantId: String) -> DownloadProgress? {
        downloadTasks[variantId]?.progress
    }
    
    /// Get download task for a variant
    /// - Parameter variantId: Variant ID
    /// - Returns: DownloadTask or nil
    public func task(for variantId: String) -> DownloadTask? {
        downloadTasks[variantId]
    }
    
    /// Get all installed variants
    public var installedVariants: [ModelVariant] {
        registry.allVariants.filter { installedVariantIds.contains($0.id) }
    }
    
    /// Get all active download tasks
    public var activeDownloads: [DownloadTask] {
        Array(downloadTasks.values).filter { $0.status.isActive }
    }
    
    /// Total size of installed models
    public var installedSize: Int64 {
        installedVariants.reduce(0) { $0 + $1.sizeBytes }
    }
    
    /// Human-readable installed size
    public var installedSizeString: String {
        ByteCountFormatter.string(fromByteCount: installedSize, countStyle: .file)
    }
    
    // MARK: - Private

    /// Scope a package operation to variants that do not already belong to an
    /// active standalone or package download. Returning nil means there is no new
    /// work; callers must not disturb the existing operations' state.
    nonisolated static func packageForDownload(
        _ package: ModelPackage,
        excluding activeVariantIds: Set<String>,
        registry: ModelCatalog
    ) -> ModelPackage? {
        let requestedIds = package.variantIds.filter { !activeVariantIds.contains($0) }
        guard !requestedIds.isEmpty else { return nil }
        let requestedSize = requestedIds.reduce(into: Int64(0)) { total, variantId in
            total += registry.variant(id: variantId)?.sizeBytes ?? 0
        }
        return ModelPackage(
            id: package.id,
            name: package.name,
            description: package.description,
            variantIds: requestedIds,
            totalSizeBytes: requestedSize
        )
    }

    /// Create UI task rows before starting package I/O. Package streams may fail
    /// during storage validation or Hub setup without yielding any progress.
    nonisolated static func packageDownloadTasks(
        for package: ModelPackage,
        registry: ModelCatalog,
        startedAt: Date
    ) -> [String: DownloadTask] {
        package.variantIds.reduce(into: [:]) { tasks, variantId in
            guard let variant = registry.variant(id: variantId) else { return }
            tasks[variantId] = DownloadTask(
                id: variantId,
                variant: variant,
                status: .downloading,
                progress: nil,
                startedAt: startedAt
            )
        }
    }

    /// Completed variants are successful independent downloads even when a later
    /// member of their package fails. Keep their installed flag, but remove their
    /// transient task entries on the package error path.
    nonisolated static func completedTaskIdsToRemoveAfterPackageFailure(
        ownedVariantIds: Set<String>,
        downloadTasks: [String: DownloadTask]
    ) -> Set<String> {
        Set(ownedVariantIds.filter { downloadTasks[$0]?.status == .completed })
    }
    
    /// Refresh list of installed models from disk
    private func refreshInstalledModels() async {
        var installed = Set<String>()
        
        for variant in registry.allVariants {
            if ModelDownloader.shared.isDownloaded(variant: variant) {
                installed.insert(variant.id)
            }
        }
        
        installedVariantIds = installed
    }
    
    /// Force refresh installed models (call after external changes)
    public func refresh() async {
        await refreshInstalledModels()
    }
}

// MARK: - Convenience for Package Checks

@available(macOS 14.0, iOS 17.0, *)
extension ModelManager {
    /// Check if all variants in a package are installed
    /// - Parameter package: Package to check
    /// - Returns: True if all variants are installed
    public func isPackageInstalled(_ package: ModelPackage) -> Bool {
        package.variantIds.allSatisfy { installedVariantIds.contains($0) }
    }
    
    /// Get install progress for a package (0.0 - 1.0)
    /// - Parameter package: Package to check
    /// - Returns: Fraction of variants installed
    public func packageProgress(_ package: ModelPackage) -> Double {
        guard !package.variantIds.isEmpty else { return 1.0 }
        let installed = package.variantIds.filter { installedVariantIds.contains($0) }.count
        return Double(installed) / Double(package.variantIds.count)
    }
    
    /// Get variants not yet installed for a package
    /// - Parameter package: Package to check
    /// - Returns: Array of variants still needed
    public func remainingVariants(for package: ModelPackage) -> [ModelVariant] {
        package.variantIds
            .filter { !installedVariantIds.contains($0) }
            .compactMap { registry.variant(id: $0) }
    }
}
