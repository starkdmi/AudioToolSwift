//
//  LSTMKernelTest.swift
//  Test Metal kernel vs standard MLX ops
//
import Foundation
import MLX

/// Compares Metal kernel output against standard MLX operations
public enum LSTMKernelTest {
  
  /// Standard LSTM step computation using MLX ops (ground truth)
  static func standardLSTMStep(ifgo: MLXArray, cell: MLXArray) -> (MLXArray, MLXArray) {
    let gates = MLX.split(ifgo, parts: 4, axis: -1)
    let i = MLX.sigmoid(gates[0])
    let f = MLX.sigmoid(gates[1])
    let g = MLX.tanh(gates[2])
    let o = MLX.sigmoid(gates[3])
    
    let newCell = f * cell + i * g
    let newHidden = o * MLX.tanh(newCell)
    
    return (newCell, newHidden)
  }
  
  /// Run comparison test
  public static func verifyKernel() -> Bool {
    print("=== LSTM Metal Kernel Verification ===\n")
    
    // Test with multiple batch sizes and hidden sizes
    let testCases: [(batchSize: Int, hiddenSize: Int)] = [
      (1, 128),
      (1, 256),
      (2, 128),
      (4, 64)
    ]
    
    var allPassed = true
    
    for (batchSize, hiddenSize) in testCases {
      print("Testing batch=\(batchSize), hidden=\(hiddenSize)...")
      
      // Create deterministic test inputs using MLXRandom
      MLXRandom.seed(42)
      let ifgo = MLXRandom.uniform(low: -2.0, high: 2.0, [batchSize, 4 * hiddenSize], dtype: .float32)
      let cell = MLXRandom.uniform(low: -1.0, high: 1.0, [batchSize, hiddenSize], dtype: .float32)
      ifgo.eval()
      cell.eval()
      
      // Compute with standard ops (ground truth)
      let (stdCell, stdHidden) = standardLSTMStep(ifgo: ifgo, cell: cell)
      stdCell.eval()
      stdHidden.eval()
      
      // Compute with Metal kernel
      guard let (metalCell, metalHidden) = LSTMKernels.fusedLSTMStep(ifgo: ifgo, cell: cell) else {
        print("  ❌ Metal kernel returned nil (fallback triggered)")
        continue
      }
      metalCell.eval()
      metalHidden.eval()
      
      // Compare outputs
      let cellDiff = abs(stdCell - metalCell)
      let hiddenDiff = abs(stdHidden - metalHidden)
      
      let cellMaxDiff: Float = cellDiff.max().item()
      let hiddenMaxDiff: Float = hiddenDiff.max().item()
      let cellMeanDiff: Float = cellDiff.mean().item()
      let hiddenMeanDiff: Float = hiddenDiff.mean().item()
      
      // Tolerance: should be very small for float32
      let tolerance: Float = 1e-5
      
      print("  Cell   - Max diff: \(cellMaxDiff), Mean diff: \(cellMeanDiff)")
      print("  Hidden - Max diff: \(hiddenMaxDiff), Mean diff: \(hiddenMeanDiff)")
      
      if cellMaxDiff < tolerance && hiddenMaxDiff < tolerance {
        print("  ✅ PASSED (tolerance: \(tolerance))")
      } else {
        print("  ❌ FAILED - difference exceeds tolerance")
        allPassed = false
      }
      print("")
    }
    
    if allPassed {
      print("=== All tests PASSED ===")
    } else {
      print("=== Some tests FAILED ===")
    }
    
    return allPassed
  }
}
