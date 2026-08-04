import MLX
import MLXNN
import Foundation

// MARK: - Spectral Processing

/// MLX implementation of Demucs spectro
/// - Parameters:
///   - x: Input audio (Batch, Channels, Time) or (Channels, Time)
///   - nFFT: Window size (FFT size will be nFFT * (1 + pad))
///   - hopLength: Stride (default: nFFT / 4)
///   - pad: Padding factor for FFT size
/// - Returns: Complex spectrogram (Batch, Channels, Freq, Frames)
public func spectro(_ x: MLXArray, nFFT: Int = 512, hopLength: Int? = nil, pad: Int = 0) -> MLXArray {
    let hop = hopLength ?? (nFFT / 4)
    
    // In Demucs, nFFT arg usually refers to the window length.
    // The actual FFT size is nFFT * (1 + pad).
    let fftSize = nFFT * (1 + pad)
    let winLength = nFFT
    
    // Create periodic Hann window
    let window = createPeriodicHannWindow(winLength: winLength)
    
    // Handle dimensions
    let shape = x.shape
    let B: Int
    let C: Int
    let T: Int
    
    if x.ndim == 3 {
        // (Batch, Channels, Time)
        B = shape[0]
        C = shape[1]
        T = shape[2]
    } else if x.ndim == 2 {
        // (Channels, Time)
        B = 1
        C = shape[0]
        T = shape[1]
    } else {
        fatalError("Expected 2D or 3D input, got \(x.ndim)")
    }
    
    // Reshape to (B*C, T) for STFT
    let xReshaped: MLXArray
    if x.ndim == 3 {
        xReshaped = x.reshaped([B * C, T])
    } else {
        xReshaped = x
    }
    
    // Call optimized MLX STFT
    let (real, imag) = stft(xReshaped, nFFT: fftSize, hopLength: hop, winLength: winLength, window: window, center: true)
    
    // real, imag are (B*C, Freq, Frames)
    
    // MLX does not support normalized=True (Unitary STFT).
    // PyTorch normalized=True means forward pass is scaled by 1/sqrt(N).
    // We must apply this manually to match the weights.
    let scale = sqrt(Float(fftSize))
    let realScaled = real / MLXArray(scale)
    let imagScaled = imag / MLXArray(scale)
    
    // Combine to complex array
    let zComplex = realScaled + MLXArray(real: 0, imaginary: 1) * imagScaled
    
    // Reshape back
    let Freq = zComplex.shape[1]
    let Frames = zComplex.shape[2]
    
    if x.ndim == 3 {
        return zComplex.reshaped([B, C, Freq, Frames])
    } else {
        return zComplex.reshaped([C, Freq, Frames])
    }
}

/// MLX implementation of Demucs ispectro
/// - Parameters:
///   - z: Complex input (Batch, Channels, Freq, Frames)
///   - hopLength: Hop length (default: derived from FFT size)
///   - length: Expected output length
///   - pad: Padding factor that was used in spectro
/// - Returns: Reconstructed audio (Batch, Channels, Time)
public func ispectro(_ z: MLXArray, hopLength: Int? = nil, length: Int? = nil, pad: Int = 0) -> MLXArray {
    // z is complex input
    
    // Derive fft_size from freq dim
    let Freq: Int
    if z.ndim == 4 {
        Freq = z.shape[2]
    } else {
        Freq = z.shape[1]
    }
    
    let fftSize = (Freq - 1) * 2
    let scale = sqrt(Float(fftSize))
    
    // Scale up to cancel forward scaling
    let zScaled = z * MLXArray(scale)
    
    let real = zScaled.realPart()
    let imag = zScaled.imaginaryPart()
    
    // Flatten batch/channels
    let B: Int
    let C: Int
    let realFlat: MLXArray
    let imagFlat: MLXArray
    
    if z.ndim == 4 {
        B = z.shape[0]
        C = z.shape[1]
        let Frames = z.shape[3]
        realFlat = real.reshaped([B * C, Freq, Frames])
        imagFlat = imag.reshaped([B * C, Freq, Frames])
    } else {
        B = 1
        C = z.shape[0]
        realFlat = real
        imagFlat = imag
    }
    
    // win_length logic
    let winLength = fftSize / (1 + pad)
    let hop = hopLength ?? (winLength / 4)
    
    let window = createPeriodicHannWindow(winLength: winLength)
    
    let xOut = istft(
        realPart: realFlat,
        imagPart: imagFlat,
        nFFT: fftSize,
        hopLength: hop,
        winLength: winLength,
        window: window,
        center: true,
        audioLength: length
    )
    
    // Reshape back
    if z.ndim == 4 {
        return xOut.reshaped([B, C, -1])
    } else {
        return xOut.reshaped([C, -1])
    }
}
