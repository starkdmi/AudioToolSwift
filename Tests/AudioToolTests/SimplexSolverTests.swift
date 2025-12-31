//
//  SimplexSolverTests.swift
//  ClearVoiceTests
//
//  Unit tests for SimplexSolver
//

import XCTest
@testable import ClearVoiceTTS

final class SimplexSolverTests: XCTestCase {
    
    let solver = SimplexSolver()
    
    // MARK: - Basic Functionality Tests
    
    func testEmptyInput() {
        let weights = solver.solve(basisVectors: [], target: [])
        XCTAssertTrue(weights.isEmpty)
    }
    
    func testSingleBasisVector() {
        // With single basis vector, weight should be 1.0
        let basis = [[1.0, 0.0, 0.0] as [Float]]
        let target: [Float] = [1.0, 0.0, 0.0]
        
        let weights = solver.solve(basisVectors: basis, target: target)
        
        XCTAssertEqual(weights.count, 1)
        XCTAssertEqual(weights[0], 1.0, accuracy: 0.01)
    }
    
    func testWeightsSumToOne() {
        // Generate random basis vectors and target
        let basis: [[Float]] = [
            [0.5, 0.3, 0.2, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.2, 0.2, 0.4, 0.2],
        ]
        let target: [Float] = [0.3, 0.4, 0.2, 0.1]
        
        let weights = solver.solve(basisVectors: basis, target: target)
        
        // Weights should sum to 1.0
        let sum = weights.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.01, "Weights should sum to 1.0")
    }
    
    func testWeightsNonNegative() {
        let basis: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        // Target that would require negative weights in unconstrained solution
        let target: [Float] = [0.8, 0.1, 0.1]
        
        let weights = solver.solve(basisVectors: basis, target: target)
        
        // All weights should be >= 0
        for (i, w) in weights.enumerated() {
            XCTAssertGreaterThanOrEqual(w, 0, "Weight \(i) should be non-negative")
        }
    }
    
    func testExactMatchWithBasisVector() {
        // Target exactly matches one basis vector
        let basis: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        let target: [Float] = [0.0, 1.0, 0.0]  // Matches second basis
        
        let weights = solver.solve(basisVectors: basis, target: target)
        
        // Second weight should be close to 1.0, others close to 0
        XCTAssertGreaterThan(weights[1], 0.9, "Second weight should be dominant")
    }
    
    func testBlendedTarget() {
        // Target is 60% first basis + 40% second basis
        let v1: [Float] = [1.0, 0.0, 0.0, 0.0]
        let v2: [Float] = [0.0, 1.0, 0.0, 0.0]
        let v3: [Float] = [0.0, 0.0, 1.0, 0.0]
        
        let basis = [v1, v2, v3]
        
        // Create target as blend: 0.6*v1 + 0.4*v2
        var target = [Float](repeating: 0, count: 4)
        for i in 0..<4 {
            target[i] = 0.6 * v1[i] + 0.4 * v2[i]
        }
        
        let weights = solver.solve(basisVectors: basis, target: target)
        
        // Should recover approximately 0.6, 0.4, 0.0
        XCTAssertEqual(weights[0], 0.6, accuracy: 0.1, "First weight should be ~0.6")
        XCTAssertEqual(weights[1], 0.4, accuracy: 0.1, "Second weight should be ~0.4")
        XCTAssertLessThan(weights[2], 0.1, "Third weight should be near 0")
    }
    
    // MARK: - Cosine Similarity Tests
    
    func testCosineSimilarityIdentical() {
        let a: [Float] = [1.0, 2.0, 3.0]
        let similarity = SimplexSolver.cosineSimilarity(a, a)
        XCTAssertEqual(similarity, 1.0, accuracy: 0.001)
    }
    
    func testCosineSimilarityOrthogonal() {
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [0.0, 1.0, 0.0]
        let similarity = SimplexSolver.cosineSimilarity(a, b)
        XCTAssertEqual(similarity, 0.0, accuracy: 0.001)
    }
    
    func testCosineSimilarityOpposite() {
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [-1.0, 0.0, 0.0]
        let similarity = SimplexSolver.cosineSimilarity(a, b)
        XCTAssertEqual(similarity, -1.0, accuracy: 0.001)
    }
    
    // MARK: - Top-K Tests
    
    func testTopKIndices() {
        let values: [Float] = [0.1, 0.9, 0.3, 0.7, 0.2]
        let top3 = SimplexSolver.topKIndices(values, k: 3)
        
        XCTAssertEqual(top3.count, 3)
        XCTAssertEqual(top3[0], 1)  // 0.9
        XCTAssertEqual(top3[1], 3)  // 0.7
        XCTAssertEqual(top3[2], 2)  // 0.3
    }
    
    func testTopKIndicesWithKGreaterThanCount() {
        let values: [Float] = [0.1, 0.9]
        let top5 = SimplexSolver.topKIndices(values, k: 5)
        
        XCTAssertEqual(top5.count, 2)
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceWith256Dim() {
        // Typical embedding dimension
        let D = 256
        let K = 10  // 10 voices
        
        // Generate random data
        var basis: [[Float]] = []
        for _ in 0..<K {
            var vec = [Float](repeating: 0, count: D)
            for j in 0..<D {
                vec[j] = Float.random(in: -1...1)
            }
            basis.append(vec)
        }
        
        var target = [Float](repeating: 0, count: D)
        for j in 0..<D {
            target[j] = Float.random(in: -1...1)
        }
        
        measure {
            _ = solver.solve(basisVectors: basis, target: target)
        }
    }
}
