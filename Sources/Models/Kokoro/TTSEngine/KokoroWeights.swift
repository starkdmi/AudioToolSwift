//
//  Kokoro-tts-lib
//
//  Weight lookup that reports a missing tensor instead of trapping
//
import Foundation
import MLX

/// Errors raised while building the model from a checkpoint.
public enum KokoroWeightError: LocalizedError {
  /// A tensor the model needs is not in the checkpoint.
  case missingWeight(key: String)
  /// The checkpoint could not be read at all.
  case unreadableCheckpoint(path: String, underlying: Error)

  public var errorDescription: String? {
    switch self {
    case .missingWeight(let key):
      return "Kokoro checkpoint is missing the weight '\(key)'."
    case .unreadableCheckpoint(let path, let underlying):
      return "Kokoro checkpoint at \(path) could not be read: \(underlying.localizedDescription)"
    }
  }
}

/// Fetches one tensor by name.
///
/// Every weight lookup in this model tree used to be a force unwrap, and the loader
/// above them a `try!`. Between them, a truncated download, a checkpoint for a
/// different Kokoro revision, or a path the caller mistyped took down the host
/// process - from inside a library, on input the caller is explicitly allowed to
/// supply (``KokoroTTSProvider`` accepts a local weights path). The initialisers
/// throw now, and the provider turns the throw into an `AudioToolError` like every
/// other load failure.
enum KokoroWeights {
  static func require(_ weights: [String: MLXArray], _ key: String) throws -> MLXArray {
    guard let value = weights[key] else {
      throw KokoroWeightError.missingWeight(key: key)
    }
    return value
  }
}
