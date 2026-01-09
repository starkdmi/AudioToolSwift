//
//  DownloadTypes.swift
//  ClearVoiceCore
//
//  Download state, progress, and error types
//

import Foundation

// MARK: - Download Error

/// Errors that can occur during model downloads
public enum DownloadError: Error, LocalizedError, Sendable {
    /// Not enough disk space available
    case insufficientStorage(required: Int64, available: Int64)
    
    /// No network connection
    case networkUnavailable
    
    /// Download was cancelled by user
    case cancelled
    
    /// Downloaded file verification failed
    case fileVerificationFailed
    
    /// Server returned error status
    case serverError(statusCode: Int)
    
    /// Model variant not found in registry
    case variantNotFound(id: String)
    
    /// Wrapped unknown error
    case unknownError(underlying: Error)
    
    public var errorDescription: String? {
        switch self {
        case .insufficientStorage(let required, let available):
            let requiredStr = ByteCountFormatter.string(fromByteCount: required, countStyle: .file)
            let availableStr = ByteCountFormatter.string(fromByteCount: available, countStyle: .file)
            return "Need \(requiredStr), only \(availableStr) available"
        case .networkUnavailable:
            return "No internet connection"
        case .cancelled:
            return "Download cancelled"
        case .fileVerificationFailed:
            return "Downloaded file is corrupted"
        case .serverError(let code):
            return "Server error (HTTP \(code))"
        case .variantNotFound(let id):
            return "Model variant '\(id)' not found"
        case .unknownError(let error):
            return error.localizedDescription
        }
    }
}

// MARK: - Download Status

/// Status of a download task
public enum DownloadStatus: Sendable, Equatable {
    /// Waiting in queue
    case queued
    
    /// Actively downloading
    case downloading
    
    /// Download paused (not yet implemented)
    case paused
    
    /// Successfully completed
    case completed
    
    /// Failed with error
    case failed(String)
    
    /// Cancelled by user
    case cancelled
    
    /// Whether download is in progress
    public var isActive: Bool {
        switch self {
        case .queued, .downloading:
            return true
        default:
            return false
        }
    }
    
    /// Whether download is finished (success or failure)
    public var isFinished: Bool {
        switch self {
        case .completed, .failed, .cancelled:
            return true
        default:
            return false
        }
    }
}

// MARK: - Download Task

/// Represents the state of a single download
public struct DownloadTask: Identifiable, Sendable {
    /// Variant ID being downloaded
    public let id: String
    
    /// Model variant being downloaded
    public let variant: ModelVariant
    
    /// Current status
    public var status: DownloadStatus
    
    /// Download progress (nil if not started)
    public var progress: DownloadProgress?
    
    /// When download started
    public var startedAt: Date?
    
    /// Error message if failed
    public var errorMessage: String?
    
    public init(
        id: String,
        variant: ModelVariant,
        status: DownloadStatus = .queued,
        progress: DownloadProgress? = nil,
        startedAt: Date? = nil,
        errorMessage: String? = nil
    ) {
        self.id = id
        self.variant = variant
        self.status = status
        self.progress = progress
        self.startedAt = startedAt
        self.errorMessage = errorMessage
    }
    
    /// Elapsed time since download started
    public var elapsedTime: TimeInterval? {
        guard let startedAt else { return nil }
        return Date().timeIntervalSince(startedAt)
    }
    
    /// Estimated time remaining based on current speed
    public var estimatedTimeRemaining: TimeInterval? {
        guard let progress, let speed = progress.bytesPerSecond, speed > 0 else {
            return nil
        }
        let remainingBytes = progress.totalBytes - progress.completedBytes
        return Double(remainingBytes) / speed
    }
}

// MARK: - Package Download Progress

/// Progress for downloading a package of multiple models
public struct PackageDownloadProgress: Sendable {
    /// Currently downloading variant
    public let currentVariant: ModelVariant
    
    /// Progress of current variant (nil if checking/preparing)
    public let variantProgress: DownloadProgress?
    
    /// Number of completed downloads
    public let completedCount: Int
    
    /// Total number of downloads
    public let totalCount: Int
    
    public init(
        currentVariant: ModelVariant,
        variantProgress: DownloadProgress?,
        completedCount: Int,
        totalCount: Int
    ) {
        self.currentVariant = currentVariant
        self.variantProgress = variantProgress
        self.completedCount = completedCount
        self.totalCount = totalCount
    }
    
    /// Overall progress fraction (0.0 - 1.0)
    public var overallFraction: Double {
        guard totalCount > 0 else { return 0 }
        let base = Double(completedCount) / Double(totalCount)
        let current = (variantProgress?.fractionCompleted ?? 0) / Double(totalCount)
        return base + current
    }
    
    /// Overall progress percentage (0-100)
    public var overallPercent: Int {
        Int(overallFraction * 100)
    }
}

// MARK: - Installed Model Info

/// Information about an installed model
public struct InstalledModel: Identifiable, Sendable {
    /// Variant ID
    public let id: String
    
    /// Model variant
    public let variant: ModelVariant
    
    /// Local path to model files
    public let localPath: URL
    
    /// Actual size on disk
    public let sizeOnDisk: Int64
    
    /// When the model was downloaded
    public let installedAt: Date?
    
    public init(
        id: String,
        variant: ModelVariant,
        localPath: URL,
        sizeOnDisk: Int64,
        installedAt: Date? = nil
    ) {
        self.id = id
        self.variant = variant
        self.localPath = localPath
        self.sizeOnDisk = sizeOnDisk
        self.installedAt = installedAt
    }
    
    /// Human-readable size string
    public var sizeString: String {
        ByteCountFormatter.string(fromByteCount: sizeOnDisk, countStyle: .file)
    }
}
