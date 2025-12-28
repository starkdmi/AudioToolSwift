//
//  MLXSTFT.swift
//  ClearVoiceCoreML
//
//  MLX-based STFT/ISTFT for MossFormer GAN SE CoreML
//  Adapted from mossformer_gan_se_coreml/mlx_stft/STFT.swift
//

import Foundation
import MLX
import MLXNN

// MARK: - Window Creation

/// Creates a periodic Hann window (matches torch.hann_window(periodic=True))
public func createPeriodicHannWindow(length: Int, dtype: DType = .float32) -> MLXArray {
    let n = MLXArray(0..<length).asType(dtype)
    let pi = MLXArray(Float.pi)
    let windowLength = MLXArray(Float(length))
    return 0.5 * (1.0 - MLX.cos(2.0 * pi * n / windowLength))
}

// MARK: - MLX STFT

/// STFT using MLX (high-performance)
/// - Parameters:
///   - x: Input signal [batch, samples] or [samples]
///   - nFFT: FFT size
///   - hopLength: Hop length between frames
///   - winLength: Window length  
///   - window: Window function
///   - center: Whether to center the signal with reflection padding
/// - Returns: Tuple of (real, imaginary) parts with shape [batch, freq, time]
public func mlxSTFT(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true
) -> (MLXArray, MLXArray) {
    let shape = x.shape
    
    guard shape.count >= 1 && (shape.count == 1 || shape[0] > 0) else {
        let emptyResult = MLXArray.zeros([1, (nFFT / 2) + 1, 1], dtype: x.dtype)
        return (emptyResult, emptyResult)
    }
    
    // Handle both 1D and 2D inputs
    let batchSize: Int
    var inputFor2D: MLXArray
    
    if shape.count == 1 {
        batchSize = 1
        inputFor2D = x.expandedDimensions(axis: 0)
    } else {
        batchSize = shape[0]
        inputFor2D = x
    }
    
    // Apply reflection padding if center
    if center {
        let padAmount = nFFT / 2
        let signalLength = inputFor2D.shape[1]
        
        // Reflection padding
        if signalLength > padAmount {
            let leftPad = inputFor2D[0..., 1..<(padAmount + 1)][0..., .stride(by: -1)]
            let rightPad = inputFor2D[0..., -(padAmount + 1)..<(-1)][0..., .stride(by: -1)]
            inputFor2D = MLX.concatenated([leftPad, inputFor2D, rightPad], axis: 1)
        } else {
            // Zero padding fallback for short signals
            let zeros = MLXArray.zeros([batchSize, padAmount], dtype: x.dtype)
            inputFor2D = MLX.concatenated([zeros, inputFor2D, zeros], axis: 1)
        }
    }
    
    let signalLength = inputFor2D.shape[1]
    let numFrames = (signalLength - winLength) / hopLength + 1
    
    if numFrames <= 0 {
        let emptyResult = MLXArray.zeros([batchSize, (nFFT / 2) + 1, 1], dtype: x.dtype)
        return (emptyResult, emptyResult)
    }
    
    // Use native asStrided for efficient frame extraction
    let frames = MLX.asStrided(
        inputFor2D,
        [batchSize, numFrames, winLength],
        strides: [signalLength, hopLength, 1]
    )
    
    // Apply window
    let windowedFrames = frames * window
    
    // Pad frames to nFFT if needed
    let finalFrames: MLXArray
    if winLength < nFFT {
        let padAmount = nFFT - winLength
        finalFrames = MLX.padded(windowedFrames, widths: [IntOrPair(0), IntOrPair(0), IntOrPair([0, padAmount])])
    } else {
        finalFrames = windowedFrames
    }
    
    // Compute FFT for all frames
    let stftComplex = rfft(finalFrames, n: nFFT, axis: -1)
    
    // Transpose to [batch, freq, time]
    let stftTransposed = stftComplex.transposed(0, 2, 1)
    
    return (stftTransposed.realPart(), stftTransposed.imaginaryPart())
}

// MARK: - MLX ISTFT

/// ISTFT using MLX (overlap-add with scatter operations)
/// - Parameters:
///   - realPart: Real part of STFT [batch, freq, time]
///   - imagPart: Imaginary part of STFT [batch, freq, time]
///   - nFFT: FFT size
///   - hopLength: Hop length
///   - winLength: Window length
///   - window: Window function
///   - center: Whether signal was centered
///   - audioLength: Target output length
/// - Returns: Reconstructed audio [batch, samples]
public func mlxISTFT(
    realPart: MLXArray,
    imagPart: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true,
    audioLength: Int? = nil
) -> MLXArray {
    // Reconstruct complex STFT
    let stftComplex = realPart + MLXArray(real: 0, imaginary: 1) * imagPart
    
    // Transpose from [batch, freq, time] to [batch, time, freq]
    let stftTransposed = stftComplex.transposed(0, 2, 1)
    
    // Get windowed time-domain frames via irfft
    let timeFrames = irfft(stftTransposed, n: nFFT, axis: -1)
    let windowedFrames = timeFrames * window
    
    let batchSize = windowedFrames.shape[0]
    let numFrames = windowedFrames.shape[1]
    let frameLength = windowedFrames.shape[2]
    let olaLen = (numFrames - 1) * hopLength + frameLength
    
    // Create normalization buffer with window squared
    let windowSquared = window * window
    var normBuffer = MLXArray.zeros([olaLen], dtype: .float32)
    
    // Pre-compute positions for scatter operations
    let positions = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                   MLXArray(0..<frameLength).expandedDimensions(axis: 0)
    let positionsFlat = positions.flattened()
    let windowSqTiled = tiled(windowSquared, repetitions: [numFrames])
    
    // Build normalization buffer
    normBuffer = normBuffer.at[positionsFlat].add(windowSqTiled)
    normBuffer = MLX.maximum(normBuffer, MLXArray(Float(1e-8)))
    
    // Overlap-add for each batch
    var output = MLXArray.zeros([batchSize, olaLen], dtype: .float32)
    let windowedFlat = windowedFrames.reshaped([batchSize, -1])
    
    for b in 0..<batchSize {
        output[b] = output[b].at[positionsFlat].add(windowedFlat[b])
    }
    
    // Normalize
    output = output / normBuffer.expandedDimensions(axis: 0)
    
    // Remove center padding
    if center {
        let trimAmount = nFFT / 2
        let resultLen = output.shape[1]
        if resultLen > 2 * trimAmount {
            output = output[0..., trimAmount..<(resultLen - trimAmount)]
        }
    }
    
    // Trim to target length
    if let audioLength = audioLength {
        let currentLen = output.shape[1]
        if currentLen > audioLength {
            output = output[0..., 0..<audioLength]
        } else if currentLen < audioLength {
            let padAmount = audioLength - currentLen
            let padding = MLXArray.zeros([output.shape[0], padAmount], dtype: output.dtype)
            output = MLX.concatenated([output, padding], axis: 1)
        }
    }
    
    return output
}
