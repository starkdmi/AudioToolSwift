//
//  FileManagerExtensions.swift
//  ClearVoiceCore
//
//  FileManager extensions for storage management
//

import Foundation

extension FileManager {
    
    /// Get available storage space on the volume
    /// - Returns: Available bytes for important usage
    /// - Throws: If resource values cannot be retrieved
    public func availableStorageSpace() throws -> Int64 {
        let homeURL = homeDirectoryForCurrentUser
        let values = try homeURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values.volumeAvailableCapacityForImportantUsage ?? 0
    }
    
    /// Calculate total size of a directory recursively
    /// - Parameter url: Directory URL
    /// - Returns: Total size in bytes
    public func directorySize(at url: URL) -> Int64 {
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
    
    /// Get modification date of a file or directory
    /// - Parameter url: File or directory URL
    /// - Returns: Modification date or nil
    public func modificationDate(at url: URL) -> Date? {
        try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
    }
}
