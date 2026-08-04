import Foundation
import MLX
import MLXNN

// MARK: - FFT-based STFT/iSTFT for FRCRN SE (exact port of Python stft_mlx.py/istft_mlx.py)

/// Creates sqrt(hann) window matching Python's create_sqrt_hann_window_mlx
/// Formula: sqrt(0.5 * (1 - cos(2π * n / N))) - periodic hann
public func createSqrtHannWindow(winLength: Int, dtype: DType = .float32) -> MLXArray {
    let n = MLXArray(0..<winLength).asType(dtype)
    let length = MLXArray(Float(winLength))
    // Periodic hann window (like scipy fftbins=True)
    let hann = 0.5 * (1 - MLX.cos(2 * Float.pi * n / length))
    return MLX.sqrt(hann)
}

// MARK: - STFT (exact port of Python stft_mlx.py)

/// Short-Time Fourier Transform matching FRCRN's approach
/// Exact port of Python stft_mlx.py stft() function
public func stftFFT(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true
) -> (real: MLXArray, imag: MLXArray) {
    // Ensure 2D input (Batch, Time)
    var signal = x
    if signal.ndim == 1 {
        signal = signal.expandedDimensions(axis: 0)
    }
    
    let batchSize = signal.shape[0]
    
    // Pad Window to n_fft if needed
    var win = window
    if winLength < nFFT {
        let pad = nFFT - winLength
        win = MLX.concatenated([win, MLXArray.zeros([pad], dtype: win.dtype)], axis: 0)
    }
    
    // Pad Signal (Reflection Padding) - only if center=True
    var xPadded: MLXArray
    if center {
        let padAmount = nFFT / 2
        let signalLen = signal.shape[1]
        // Left reflection: x[:, 1:pad_amount + 1][:, ::-1]
        let leftEnd = min(padAmount + 1, signalLen)
        let leftReflect = signal[0..., 1..<leftEnd][0..., .stride(by: -1)]
        // Right reflection: x[:, -pad_amount - 1:-1][:, ::-1]
        let rightStart = max(0, signalLen - padAmount - 1)
        let rightEnd = max(0, signalLen - 1)
        let rightReflect = signal[0..., rightStart..<rightEnd][0..., .stride(by: -1)]
        xPadded = MLX.concatenated([leftReflect, signal, rightReflect], axis: -1)
    } else {
        xPadded = signal
    }
    
    // Calculate frames
    let paddedLen = xPadded.shape[xPadded.ndim - 1]
    let numFrames = (paddedLen - winLength) / hopLength + 1
    
    // Ensure signal is long enough for striding
    let targetLen = (numFrames - 1) * hopLength + nFFT
    
    if paddedLen < targetLen {
        let padExtra = targetLen - paddedLen
        xPadded = MLX.concatenated([xPadded, MLXArray.zeros([batchSize, padExtra], dtype: signal.dtype)], axis: -1)
    }
    
    // Create strided views for each frame
    let shape = [batchSize, numFrames, nFFT]
    let strides = [xPadded.shape[xPadded.ndim - 1], hopLength, 1]
    
    let frames = MLX.asStrided(xPadded, shape, strides: strides)
    
    // Apply window and compute FFT
    let stftComplex = MLX.rfft(frames * win, n: nFFT, axis: -1)
    
    // Transpose to (Batch, Freq, Time)
    let stftTransposed = stftComplex.transposed(0, 2, 1)
    
    return (stftTransposed.realPart(), stftTransposed.imaginaryPart())
}

// MARK: - iSTFT (exact port of Python istft_mlx.py)

/// Inverse Short-Time Fourier Transform matching FRCRN's approach
/// Exact port of Python istft_mlx.py istft() function
public func istftFFT(
    real: MLXArray,
    imag: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true,
    audioLength: Int? = nil
) -> MLXArray {
    // Robust Window Padding (Safety Check)
    var win = window
    if win.shape[0] < nFFT {
        let pad = nFFT - win.shape[0]
        win = MLX.concatenated([win, MLXArray.zeros([pad], dtype: win.dtype)], axis: 0)
    }
    
    // Inverse FFT
    let stftComplex = real + MLXArray(real: 0, imaginary: 1) * imag
    let timeFrames = MLX.irfft(stftComplex.transposed(0, 2, 1), n: nFFT, axis: -1)
    
    // Apply synthesis window
    let windowedFrames = timeFrames * win
    
    let batchSize = windowedFrames.shape[0]
    let numFrames = windowedFrames.shape[1]
    let frameLength = windowedFrames.shape[2]
    let olaLen = (numFrames - 1) * hopLength + frameLength
    
    // Vectorized Overlap-Add (The Speedup)
    // Allocate flat output buffer (Batch * Time) - matching Python exactly
    var output = MLXArray.zeros([batchSize * olaLen], dtype: .float32)
    
    // Calculate indices for a SINGLE frame sequence
    // positions = mx.arange(num_frames)[:, None] * hop_length + mx.arange(frame_length)[None, :]
    let frameIndices = MLXArray(0..<numFrames).expandedDimensions(axis: 1)  // [numFrames, 1]
    let sampleIndices = MLXArray(0..<frameLength).expandedDimensions(axis: 0)  // [1, frameLength]
    let positions = frameIndices * hopLength + sampleIndices  // [numFrames, frameLength]
    let positionsFlat = positions.flattened()  // [numFrames * frameLength]
    
    // Calculate global indices for ALL batches at once
    // batch_offsets = mx.arange(batch_size) * ola_len
    // global_indices = positions_flat[None, :] + batch_offsets[:, None]
    let batchOffsets = MLXArray(0..<batchSize) * olaLen  // [batchSize]
    let globalIndices = positionsFlat.expandedDimensions(axis: 0) + batchOffsets.expandedDimensions(axis: 1)
    // globalIndices: [batchSize, numFrames * frameLength]
    
    // Single scatter_add operation for the entire batch
    output = output.at[globalIndices.flattened()].add(windowedFrames.flattened())
    
    // Reshape back to (Batch, Time)
    output = output.reshaped([batchSize, olaLen])
    
    // Normalization
    // window_squared = window ** 2
    let windowSquared = win * win
    var normBuffer = MLXArray.zeros([olaLen], dtype: .float32)
    // window_sq_tiled = mx.tile(window_squared, num_frames)
    let windowSqTiled = tiled(windowSquared, repetitions: [numFrames])
    normBuffer = normBuffer.at[positionsFlat].add(windowSqTiled)
    
    // Avoid division by zero
    normBuffer = MLX.maximum(normBuffer, MLXArray(Float(1e-10)))
    output = output / normBuffer.expandedDimensions(axis: 0)
    
    // Final Trimming
    if center {
        let startCut = nFFT / 2
        output = output[0..., startCut...]
    }
    
    if let length = audioLength {
        output = output[0..., 0..<length]
    }
    
    return output
}

// MARK: - ConvSTFT Module (wrapper for FRCRN)

/// FFT-based STFT wrapper class for FRCRN MLX
/// Uses sqrt(hann) window for COLA normalization
public class ConvSTFT: Module {
    let fftLen: Int
    let winInc: Int  // hop_length
    let winLen: Int
    let window: MLXArray
    
    // Dummy weight for safetensors loading (not used in FFT-based approach)
    @ModuleInfo var weight: MLXArray
    
    public init(winLen: Int, winInc: Int, fftLen: Int) {
        self.fftLen = fftLen
        self.winInc = winInc
        self.winLen = winLen
        
        // Create sqrt(hann) window for FRCRN compatibility
        self.window = createSqrtHannWindow(winLength: winLen)
        
        // Dummy weight for safetensors loading (not used)
        let freqBins = fftLen / 2 + 1
        self._weight.wrappedValue = MLXArray.zeros([2 * freqBins, winLen, 1])
        
        super.init()
    }
    
    public func callAsFunction(_ inputs: MLXArray) -> (MLXArray, MLXArray) {
        // Squeeze batch dimension if 3D input (batch, length, 1)
        var x = inputs
        if x.ndim == 3 {
            x = x.squeezed(axis: -1)
        }
        
        // Use FFT-based STFT with center=false for FRCRN
        return stftFFT(x, nFFT: fftLen, hopLength: winInc, winLength: winLen,
                       window: window, center: false)
    }
}

// MARK: - ConviSTFT Module (wrapper for FRCRN)

/// FFT-based iSTFT wrapper class for FRCRN MLX
/// Uses sqrt(hann) window for COLA normalization
public class ConviSTFT: Module {
    let fftLen: Int
    let winInc: Int  // hop_length
    let winLen: Int
    let window: MLXArray
    
    // Dummy weights for safetensors loading (not used in FFT-based approach)
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var window_sum: MLXArray
    
    public init(winLen: Int, winInc: Int, fftLen: Int) {
        self.fftLen = fftLen
        self.winInc = winInc
        self.winLen = winLen
        
        // Create sqrt(hann) window for FRCRN compatibility
        self.window = createSqrtHannWindow(winLength: winLen)
        
        // Dummy weights for safetensors loading (not used)
        let freqBins = fftLen / 2 + 1
        self._weight.wrappedValue = MLXArray.zeros([2 * freqBins, winLen, 1])
        self._window_sum.wrappedValue = MLXArray.zeros([1])
        
        super.init()
    }
    
    public func callAsFunction(_ real: MLXArray, _ imag: MLXArray) -> MLXArray {
        // Use FFT-based iSTFT with center=false for FRCRN
        let output = istftFFT(real: real, imag: imag,
                              nFFT: fftLen, hopLength: winInc, winLength: winLen,
                              window: window, center: false, audioLength: nil)
        
        // Return with channel dimension: (batch, time, 1)
        return output.expandedDimensions(axis: -1)
    }
}
