//
//  ManagedModel.swift
//  AudioToolCore
//
//  Protocol for models that support memory lifecycle management
//

import Foundation

// MARK: - ManagedModel Protocol

/// Protocol for models that support memory lifecycle management.
///
/// Conform to this protocol to enable:
/// - Memory tracking by `ModelCatalog`
/// - LRU-based automatic eviction under memory pressure
/// - Explicit unload support
///
/// Example:
/// ```swift
/// extension MossFormer2SE48KProvider: ManagedModel {
///     public nonisolated var modelId: String { "mossformer2_se_48k" }
///     
///     public nonisolated var estimatedMemoryBytes: Int {
///         precision == .fp16 ? 200_000_000 : 400_000_000
///     }
///     
///     public func checkIfLoaded() async -> Bool { pipeline != nil }
///     
///     public func unload() async {
///         pipeline = nil
///         // GPU.clearCache() if using MLX
///     }
/// }
/// ```
public protocol ManagedModel: Sendable {
    /// Unique identifier for this model instance.
    ///
    /// Should be stable across app launches (e.g., "mossformer2_se_48k").
    /// Used as the key in `ModelCatalog`.
    var modelId: String { get }
    
    /// Estimated memory footprint in bytes (VRAM + RAM).
    ///
    /// This is an approximation used for:
    /// - Deciding when to evict models
    /// - Reporting total memory usage
    ///
    /// For MLX models, include GPU memory. For CoreML, include ANE/CPU allocations.
    var estimatedMemoryBytes: Int { get }
    
    /// Load model into memory.
    ///
    /// Called by `ModelCatalog.register(_:)` if the model isn't already loaded.
    /// Implementations should:
    /// - Download weights if needed
    /// - Load into GPU/ANE memory
    /// - Prepare for inference
    func load() async throws
    
    /// Unload model from memory (release resources).
    ///
    /// Called by `ModelCatalog.unload(modelId:)` or during LRU eviction.
    /// Implementations should:
    /// - Release model weights
    /// - Clear GPU cache if applicable
    /// - Set internal state to unloaded
    func unload() async
    
    /// Whether the model is currently loaded and ready for inference.
    ///
    /// This is an async method to support actor-isolated implementations.
    func checkIfLoaded() async -> Bool
}

// MARK: - Default Implementations

extension ManagedModel {
    /// Default memory estimate (100MB) - override for accuracy
    public var estimatedMemoryBytes: Int { 100_000_000 }
}
