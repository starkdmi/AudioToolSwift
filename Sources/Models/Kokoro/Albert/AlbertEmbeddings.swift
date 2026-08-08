//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertEmbeddings: Module {
  @ModuleInfo var wordEmbeddings: Embedding
  @ModuleInfo var positionEmbeddings: Embedding
  @ModuleInfo var tokenTypeEmbeddings: Embedding
  @ModuleInfo var layerNorm: LayerNormInference

  init(weights: [String: MLXArray], config: AlbertModelArgs) throws {
    self._wordEmbeddings.wrappedValue = Embedding(
      weight: try weights.required("bert.embeddings.word_embeddings.weight")
    )
    self._positionEmbeddings.wrappedValue = Embedding(
      weight: try weights.required("bert.embeddings.position_embeddings.weight")
    )
    self._tokenTypeEmbeddings.wrappedValue = Embedding(
      weight: try weights.required("bert.embeddings.token_type_embeddings.weight")
    )
    
    let weightName = "bert.embeddings.LayerNorm.weight"
    let biasName = "bert.embeddings.LayerNorm.bias"
    let layerNormWeights = try weights.required(weightName)
    let layerNormBiases = try weights.required(biasName)

    guard layerNormWeights.count == config.embeddingSize else {
      throw KokoroModelLoadingError.invalidWeightShape(
        name: weightName,
        expected: config.embeddingSize,
        found: layerNormWeights.count
      )
    }
    guard layerNormBiases.count == config.embeddingSize else {
      throw KokoroModelLoadingError.invalidWeightShape(
        name: biasName,
        expected: config.embeddingSize,
        found: layerNormBiases.count
      )
    }

    // Use LayerNormInference which accepts weights directly in init
    self._layerNorm.wrappedValue = LayerNormInference(weight: layerNormWeights, bias: layerNormBiases, eps: config.layerNormEps)
  }

  func callAsFunction(
    _ inputIds: MLXArray,
    tokenTypeIds: MLXArray? = nil,
    positionIds: MLXArray? = nil
  ) -> MLXArray {
    let seqLength = inputIds.shape[1]

    let positionIdsUsed: MLXArray
    if let positionIds = positionIds {
      positionIdsUsed = positionIds
    } else {
      positionIdsUsed = MLX.expandedDimensions(MLXArray(0 ..< seqLength), axes: [0])
    }

    let tokenTypeIdsUsed: MLXArray
    if let tokenTypeIds = tokenTypeIds {
      tokenTypeIdsUsed = tokenTypeIds
    } else {
      tokenTypeIdsUsed = MLXArray.zeros(like: inputIds)
    }

    let wordsEmbeddings = wordEmbeddings(inputIds)
    let positionEmbeddingsResult = positionEmbeddings(positionIdsUsed)
    let tokenTypeEmbeddingsResult = tokenTypeEmbeddings(tokenTypeIdsUsed)
    var embeddings = wordsEmbeddings + positionEmbeddingsResult + tokenTypeEmbeddingsResult
    embeddings = layerNorm(embeddings)
    return embeddings
  }
}
