import Foundation
import MLX

/// Embedding loader for USS conditioning vectors
public class EmbeddingLoader {
    
    /// Available embedding types
    public enum EmbeddingType: String, CaseIterable, Sendable {
        case speech = "speech"
        case music = "music"
        case noise = "noise"
        case nature = "nature"
        case things = "things"
        case animal = "animal"
        case human = "human"
        
        var filename: String {
            return "\(rawValue)_embedding_527d.safetensors"
        }
    }
    
    /// Load embedding from .safetensors file
    public static func loadEmbedding(type: EmbeddingType, from directory: String) throws -> MLXArray {
        let path = "\(directory)/\(type.filename)"
        let url = URL(fileURLWithPath: path)
        
        // Load safetensors file
        let arrays = try MLX.loadArrays(url: url)
        
        // Get the embedding array by key
        guard let embedding = arrays["embedding"] else {
            throw EmbeddingError.noArrayFound
        }
        
        // Ensure it's the expected shape (527,)
        guard embedding.shape.count == 1 && embedding.shape[0] == 527 else {
            throw EmbeddingError.unexpectedShape(embedding.shape)
        }
        
        // Add batch dimension
        return embedding.reshaped([1, 527])
    }
}

// MARK: - Error Types

public enum EmbeddingError: Error, LocalizedError {
    case fileNotFound(String)
    case noArrayFound
    case unexpectedShape([Int])
    
    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Embedding file not found: \(path)"
        case .noArrayFound:
            return "No array found in numpy file"
        case .unexpectedShape(let shape):
            return "Unexpected embedding shape: \(shape). Expected (527,)"
        }
    }
}