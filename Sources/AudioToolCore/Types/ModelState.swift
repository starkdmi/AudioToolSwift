//
//  ModelState.swift
//  ClearVoiceCore
//
//  Model loading state for UI binding
//

import Foundation

/// Model loading state for UI binding
///
/// Use this to track and display download/loading progress in your UI:
/// ```swift
/// switch modelState {
/// case .notLoaded: Text("Tap to load")
/// case .downloading(let progress): ProgressView(value: progress)
/// case .loading: ProgressView().progressViewStyle(.circular)
/// case .ready: Text("Ready ✓")
/// case .failed(let error): Text("Error: \(error)")
/// }
/// ```
public enum ModelState: Sendable, Equatable {
    /// Model not loaded
    case notLoaded
    
    /// Downloading model weights (progress: 0.0 - 1.0)
    case downloading(progress: Double)
    
    /// Weights downloaded, initializing model
    case loading
    
    /// Model ready for inference
    case ready
    
    /// Loading failed
    case failed(String)
    
    /// Progress percentage (0-100) for downloading state
    public var progressPercent: Int? {
        if case .downloading(let progress) = self {
            return Int(progress * 100)
        }
        return nil
    }
    
    /// Whether the model is ready for inference
    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}
