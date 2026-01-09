//
//  ModelManager.swift
//  ClearVoice
//
//  SwiftUI-ready observable model manager for download UI
//

import Foundation
import ClearVoiceCore

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
    public let registry: ModelRegistry
    
    /// Last error encountered
    public private(set) var lastError: Error?
    
    /// Whether initial scan is complete
    public private(set) var isInitialized: Bool = false
    
    private init() {
        self.registry = ModelRegistry.shared
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
        for variantId in package.variantIds {
            if let variant = registry.variant(id: variantId),
               !installedVariantIds.contains(variantId) {
                download(variant)
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
    
    /// Refresh list of installed models from disk
    private func refreshInstalledModels() async {
        var installed = Set<String>()
        
        for variant in registry.allVariants {
            if ModelDownloader.shared.isDownloaded(repo: variant.repo) {
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
