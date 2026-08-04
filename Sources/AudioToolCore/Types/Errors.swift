//
//  Errors.swift
//  AudioTool
//
//  Framework error types
//

import Foundation

/// AudioTool framework errors
public enum AudioToolError: Error, Sendable {
    
    // MARK: - Model Errors
    
    /// Model not found at expected location
    case modelNotFound(String)
    
    /// Model failed to load
    case modelLoadFailed(String, underlying: Error)
    
    /// Model version incompatible
    case incompatibleModelVersion(expected: String, found: String)
    
    /// Model not loaded yet
    case modelNotLoaded(String)
    
    // MARK: - Audio Errors
    
    /// Invalid audio format
    case invalidAudioFormat(expected: String, found: String)
    
    /// Sample rate mismatch
    case sampleRateMismatch(expected: Int, found: Int)
    
    /// Empty audio buffer
    case emptyAudioBuffer
    
    /// Audio file not found
    case audioFileNotFound(URL)
    
    /// Audio loading failed
    case audioLoadFailed(URL, underlying: Error)
    
    // MARK: - Pipeline Errors
    
    /// Invalid pipeline configuration
    case pipelineConfigurationInvalid(String)
    
    /// Pipeline stage failed
    case stageFailed(stage: String, underlying: Error)
    
    /// Operation cancelled
    case cancelled
    
    // MARK: - Resource Errors
    
    /// Memory exhausted
    case memoryExhausted(required: Int, available: Int)
    
    /// Backpressure timeout
    case backpressureTimeout
    
    /// Resource not available
    case resourceUnavailable(String)
}

// MARK: - LocalizedError

extension AudioToolError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let model):
            return "Model not found: \(model)"
        case .modelLoadFailed(let model, let error):
            return "Failed to load model '\(model)': \(error.localizedDescription)"
        case .incompatibleModelVersion(let expected, let found):
            return "Incompatible model version. Expected: \(expected), Found: \(found)"
        case .modelNotLoaded(let model):
            return "Model not loaded: \(model). Call preload() first."
        case .invalidAudioFormat(let expected, let found):
            return "Invalid audio format. Expected: \(expected), Found: \(found)"
        case .sampleRateMismatch(let expected, let found):
            return "Sample rate mismatch. Expected: \(expected)Hz, Found: \(found)Hz"
        case .emptyAudioBuffer:
            return "Audio buffer is empty"
        case .audioFileNotFound(let url):
            return "Audio file not found: \(url.lastPathComponent)"
        case .audioLoadFailed(let url, let error):
            return "Failed to load audio from '\(url.lastPathComponent)': \(error.localizedDescription)"
        case .pipelineConfigurationInvalid(let reason):
            return "Invalid pipeline configuration: \(reason)"
        case .stageFailed(let stage, let error):
            return "Pipeline stage '\(stage)' failed: \(error.localizedDescription)"
        case .cancelled:
            return "Operation was cancelled"
        case .memoryExhausted(let required, let available):
            return "Memory exhausted. Required: \(required / 1_000_000)MB, Available: \(available / 1_000_000)MB"
        case .backpressureTimeout:
            return "Backpressure timeout - pipeline is congested"
        case .resourceUnavailable(let resource):
            return "Resource unavailable: \(resource)"
        }
    }
}
