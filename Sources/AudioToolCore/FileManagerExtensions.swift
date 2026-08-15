//
//  FileManagerExtensions.swift
//  AudioToolCore
//
//  FileManager extensions for storage management
//

import Foundation

/// Internal on purpose, all of it.
///
/// `availableStorageSpace()` touches one of Apple's required-reason APIs, and
/// `PrivacyInfo.xcprivacy` declares the reason that describes how *this package*
/// uses it: a disk check that decides whether a model download may proceed
/// (E174.1). Exporting it would make this package a wrapper offering that API to
/// the host app - a different declaration, and a promise about call sites we do
/// not control.
///
/// The timestamp side of that manifest is not here: it is ModelDownloader reading
/// modification dates on its own cache. A `modificationDate(at:)` taking any URL
/// used to be public in this file with no caller at all, which was exactly the
/// wrapper case, and it is gone.
extension FileManager {

    /// The home directory the model caches hang off, on every platform this package
    /// declares support for.
    ///
    /// `homeDirectoryForCurrentUser` is macOS-only - referencing it at all is a hard
    /// error under an iOS SDK, which is what kept this package from compiling for
    /// the iOS 18 it advertises. On iOS the app container root is the analogue, and
    /// the two cache layouts land in the right place under it: `Documents/huggingface`
    /// is exactly where `HubApi` puts its downloads (it resolves
    /// `.documentDirectory`, which is the container's `Documents`), and `.cache`
    /// has no OS meaning there but is still a valid, writable place for the Python
    /// -style layout the desktop tooling produces.
    var userHomeDirectory: URL {
        #if os(macOS)
        return homeDirectoryForCurrentUser
        #else
        return URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        #endif
    }

    /// Get available storage space on the volume
    /// - Returns: Available bytes for important usage
    /// - Throws: If resource values cannot be retrieved
    func availableStorageSpace() throws -> Int64 {
        let homeURL = userHomeDirectory
        let values = try homeURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values.volumeAvailableCapacityForImportantUsage ?? 0
    }
    
    /// Calculate total size of a directory recursively
    /// - Parameter url: Directory URL
    /// - Returns: Total size in bytes
    func directorySize(at url: URL) -> Int64 {
        guard let enumerator = enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
            options: [.skipsHiddenFiles],
            errorHandler: nil
        ) else {
            return 0
        }
        
        var totalSize: Int64 = 0
        
        for case let fileURL as URL in enumerator {
            guard let resourceValues = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey]),
                  resourceValues.isDirectory == false,
                  let fileSize = resourceValues.fileSize else {
                continue
            }
            totalSize += Int64(fileSize)
        }
        
        return totalSize
    }
}
