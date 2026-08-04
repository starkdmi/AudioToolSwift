//
//  SimplexSolver.swift
//  AudioToolTTS
//
//  vDSP-accelerated simplex-constrained least squares solver
//  for voice weight optimization
//

import Foundation
import Accelerate

// MARK: - Simplex Solver

/// Simplex-constrained least squares solver using projected gradient descent
///
/// Solves: min ||A @ w - b||²  subject to: w >= 0, sum(w) = 1
///
/// Used for voice matching to find optimal blend weights that minimize
/// the distance between a reference speaker embedding and a weighted
/// combination of known voice embeddings.
///
/// ## Usage
/// ```swift
/// let solver = SimplexSolver()
///
/// // Voice embeddings [K voices, D dimensions]
/// let basisVectors: [[Float]] = [embedding1, embedding2, embedding3]
///
/// // Reference embedding [D dimensions]
/// let target: [Float] = referenceEmbedding
///
/// // Solve for weights
/// let weights = solver.solve(basisVectors: basisVectors, target: target)
/// // weights.sum() ≈ 1.0, all weights >= 0
/// ```
public struct SimplexSolver: Sendable {
    
    /// Learning rate for gradient descent
    private let learningRate: Float
    
    /// Maximum iterations
    private let maxIterations: Int
    
    /// Convergence tolerance
    private let tolerance: Float
    
    /// Initialize solver with parameters
    /// - Parameters:
    ///   - learningRate: Step size for gradient descent (default: 0.1)
    ///   - maxIterations: Maximum iterations (default: 100)
    ///   - tolerance: Convergence tolerance (default: 1e-6)
    public init(
        learningRate: Float = 0.1,
        maxIterations: Int = 100,
        tolerance: Float = 1e-6
    ) {
        self.learningRate = learningRate
        self.maxIterations = maxIterations
        self.tolerance = tolerance
    }
    
    /// Solve simplex-constrained least squares problem
    ///
    /// Finds weights w that minimize ||A @ w - b||² where:
    /// - w >= 0 (non-negative)
    /// - sum(w) = 1 (probability simplex)
    ///
    /// - Parameters:
    ///   - basisVectors: Matrix of basis vectors [K, D] - K voices with D-dim embeddings
    ///   - target: Target vector [D] - reference embedding to match
    /// - Returns: Weight vector [K] on the probability simplex
    public func solve(
        basisVectors: [[Float]],
        target: [Float]
    ) -> [Float] {
        let K = basisVectors.count
        guard K > 0 else { return [] }
        
        let D = target.count
        guard D > 0 else { return Array(repeating: 1.0 / Float(K), count: K) }
        
        // Validate dimensions
        for vec in basisVectors {
            precondition(vec.count == D, "All basis vectors must have dimension \(D)")
        }
        
        // Initialize weights uniformly
        var weights = [Float](repeating: 1.0 / Float(K), count: K)
        
        // Flattened basis matrix [K * D] in row-major order
        var basisFlat = [Float](repeating: 0, count: K * D)
        for (i, vec) in basisVectors.enumerated() {
            for (j, val) in vec.enumerated() {
                basisFlat[i * D + j] = val
            }
        }
        
        // Preallocate working arrays
        var prediction = [Float](repeating: 0, count: D)
        var residual = [Float](repeating: 0, count: D)
        var gradient = [Float](repeating: 0, count: K)
        
        var previousLoss: Float = .infinity
        
        for _ in 0..<maxIterations {
            // Compute prediction: A @ w = sum(w_i * basis_i)
            vDSP_vclr(&prediction, 1, vDSP_Length(D))
            
            basisFlat.withUnsafeBufferPointer { basisPtr in
                for i in 0..<K {
                    var w = weights[i]
                    let rowPtr = basisPtr.baseAddress! + i * D
                    vDSP_vsma(
                        rowPtr, 1,
                        &w,
                        prediction, 1,
                        &prediction, 1,
                        vDSP_Length(D)
                    )
                }
            }
            
            // Compute residual: prediction - target
            vDSP_vsub(target, 1, prediction, 1, &residual, 1, vDSP_Length(D))
            
            // Compute loss: ||residual||²
            var loss: Float = 0
            vDSP_dotpr(residual, 1, residual, 1, &loss, vDSP_Length(D))
            
            // Check convergence
            if abs(previousLoss - loss) < tolerance {
                break
            }
            previousLoss = loss
            
            // Compute gradient: 2 * A^T @ residual
            // gradient[i] = 2 * dot(basis_i, residual)
            basisFlat.withUnsafeBufferPointer { basisPtr in
                for i in 0..<K {
                    var dot: Float = 0
                    let rowPtr = basisPtr.baseAddress! + i * D
                    vDSP_dotpr(
                        rowPtr, 1,
                        residual, 1,
                        &dot,
                        vDSP_Length(D)
                    )
                    gradient[i] = 2 * dot
                }
            }
            
            // Gradient step: w = w - lr * gradient
            var negLR = -learningRate
            vDSP_vsma(gradient, 1, &negLR, weights, 1, &weights, 1, vDSP_Length(K))
            
            // Project onto simplex
            projectOntoSimplex(&weights)
        }
        
        return weights
    }
    
    /// Project a vector onto the probability simplex
    ///
    /// Uses the algorithm from "Efficient Projections onto the ℓ1-Ball
    /// for Learning in High Dimensions" (Duchi et al., 2008)
    ///
    /// - Parameter v: Vector to project (modified in place)
    private func projectOntoSimplex(_ v: inout [Float]) {
        let n = v.count
        guard n > 0 else { return }
        
        // Step 1: Sort in descending order
        var sorted = v
        vDSP_vsort(&sorted, vDSP_Length(n), -1)  // -1 for descending
        
        // Step 2: Find the threshold τ
        var cumulativeSum: Float = 0
        var rho = 0
        
        for j in 0..<n {
            cumulativeSum += sorted[j]
            let test = sorted[j] - (cumulativeSum - 1) / Float(j + 1)
            if test > 0 {
                rho = j
            }
        }
        
        // Compute threshold
        var sumUpToRho: Float = 0
        for j in 0...rho {
            sumUpToRho += sorted[j]
        }
        let tau = (sumUpToRho - 1) / Float(rho + 1)
        
        // Step 3: Apply threshold: max(v - τ, 0)
        var negTau = -tau
        vDSP_vsadd(v, 1, &negTau, &v, 1, vDSP_Length(n))
        
        var zero: Float = 0
        vDSP_vthres(v, 1, &zero, &v, 1, vDSP_Length(n))
    }
}

// MARK: - Vector Utilities

extension SimplexSolver {
    
    /// Compute cosine similarity between two vectors using vDSP
    /// - Returns: Similarity in range [-1, 1]
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have same dimension")
        let n = vDSP_Length(a.count)
        guard n > 0 else { return 0 }
        
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        vDSP_dotpr(a, 1, b, 1, &dot, n)
        vDSP_dotpr(a, 1, a, 1, &normA, n)
        vDSP_dotpr(b, 1, b, 1, &normB, n)
        
        let denominator = sqrt(normA * normB)
        guard denominator > 1e-10 else { return 0 }
        
        return dot / denominator
    }
    
    /// Find indices of top-K values by descending order
    /// - Parameters:
    ///   - values: Array of values
    ///   - k: Number of top values to return
    /// - Returns: Indices of top-K values
    public static func topKIndices(_ values: [Float], k: Int) -> [Int] {
        let k = min(k, values.count)
        guard k > 0 else { return [] }
        
        // Create (index, value) pairs and sort by value descending
        let indexed = values.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        
        return sorted.prefix(k).map { $0.0 }
    }
}
