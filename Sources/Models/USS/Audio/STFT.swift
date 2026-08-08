import Foundation
import MLX
import MLXNN

// Warning: Local stft/istft is more optimized for USS than AudioUtils versions.

/// Calculate magnitude and phase from real and imaginary parts.
///
/// - Parameters:
///   - real: Real part
///   - imag: Imaginary part
/// - Returns: Tuple containing magnitude, cosine of phase, and sine of phase
public func magphase(real: MLXArray, imag: MLXArray) -> (mag: MLXArray, cos: MLXArray, sin: MLXArray) {
    let mag = MLX.sqrt(real * real + imag * imag)
    
    // Avoid division by zero
    let eps: Float = 1e-10
    let cos = real / (mag + eps)
    let sin = imag / (mag + eps)
    
    return (mag, cos, sin)
}

/// Create a Hann window in MLX.
public func createHannWindow(_ winLength: Int, periodic: Bool = false) -> MLXArray {
    let indices = MLXArray(0..<winLength)//.map { Float($0) })
    if periodic {
        return 0.5 * (1 - MLX.cos(2 * Float.pi * indices / Float(winLength)))
    } else {
        return 0.5 * (1 - MLX.cos(2 * Float.pi * indices / Float(winLength - 1)))
    }
}

/// Ultra-fast STFT with maximum optimizations:
/// - Minimal memory allocations
/// - Vectorized operations
/// - Optimized for repeated calls
/*public func mlxSTFT(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true
) -> (real: MLXArray, imag: MLXArray) {
    let batchSize = x.shape[0]
    let signalLen = x.shape[1]
    
    // 1. Reflection padding (compilation-friendly)
    
    let xPadded: MLXArray
    let paddedLen: Int
    
    if center {
        let padAmount = nFFT / 2
        // Manual reflection padding for MLX compatibility
        let leftPad = x[0..., 1..<(padAmount + 1)][0..., .stride(by: -1)]
        let rightPad = x[0..., -(padAmount + 1)..<(-1)][0..., .stride(by: -1)]
        xPadded = MLX.concatenated([leftPad, x, rightPad], axis: -1)
        paddedLen = signalLen + 2 * padAmount
    } else {
        xPadded = x
        paddedLen = signalLen
    }
    
    // 2. Single-shot framing
    let numFrames = (paddedLen - winLength) / hopLength + 1
    
    // Create frames using asStrided like the optimized version
    let frames = MLX.asStrided(
        xPadded,
        [batchSize, numFrames, winLength],
        strides: [paddedLen, hopLength, 1]
    )
    
    // 3. Apply window and handle FFT size
    let windowedFrames = frames * window[0..<winLength]
    
    let finalFrames: MLXArray
    if winLength < nFFT {
        // Pad frames to n_fft
        let padWidth = [IntOrPair(0), IntOrPair(0), IntOrPair([0, nFFT - winLength])]
        finalFrames = MLX.padded(windowedFrames, widths: padWidth)
    } else {
        finalFrames = windowedFrames
    }
    
    // 4. FFT with immediate transpose for memory efficiency
    var stftComplex = MLXFFT.rfft(finalFrames, n: nFFT, axis: -1)
    
    
    stftComplex = stftComplex.transposed(0, 2, 1)
    
    let realPart = stftComplex.realPart()
    let imagPart = stftComplex.imaginaryPart()
    
    return (realPart, imagPart)
}*/

/// Highly optimized pure MLX iSTFT with vectorized operations
/*public func mlxISTFT( // TODO: THIS IS POTTENTIALLY BAD
    realPart: MLXArray,
    imagPart: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true,
    audioLength: Int? = nil
) -> MLXArray {
    // Step 1: Get windowed time-domain frames
    let stftComplex = realPart + imagPart * MLXArray(real: 0, imaginary: 1)
    
    let timeFrames = MLXFFT.irfft(stftComplex.transposed(0, 2, 1), n: nFFT, axis: -1)
    
    // Get the actual window (may be padded)
    let windowFull: MLXArray
    if winLength < nFFT {
        let padLeft = (nFFT - winLength) / 2
        let padRight = nFFT - winLength - padLeft
        windowFull = MLX.padded(window, widths: [IntOrPair([padLeft, padRight])])
    } else {
        windowFull = window
    }
    
    let windowedFrames = timeFrames * windowFull
    
    let batchSize = windowedFrames.shape[0]
    let numFrames = windowedFrames.shape[1]
    let frameLength = windowedFrames.shape[2]
    let olaLen = (numFrames - 1) * hopLength + frameLength
    
    // Step 2: Optimized overlap-add using fold operation
    // Pad windowed frames to full output length for each position
    var outputFrames = MLX.zeros([batchSize, numFrames, olaLen], dtype: .float32)
    
    // Fill in the frames at their correct positions
    for i in 0..<numFrames {
        let start = i * hopLength
        let end = min(start + frameLength, olaLen)
        let len = end - start
        if len > 0 {
            outputFrames[0..., i, start..<end] = windowedFrames[0..., i, 0..<len]
        }
    }
    
    // Sum across frames dimension to get overlap-added result
    let output = MLX.sum(outputFrames, axis: 1)
    
    // Step 3: Create normalization buffer using same approach
    let windowSquared = windowFull * windowFull
    var normFrames = MLX.zeros([numFrames, olaLen], dtype: .float32)
    
    for i in 0..<numFrames {
        let start = i * hopLength
        let end = min(start + frameLength, olaLen)
        let len = end - start
        if len > 0 {
            normFrames[i, start..<end] = windowSquared[0..<len]
        }
    }
    
    // Sum to get normalization buffer
    let normBuffer = MLX.sum(normFrames, axis: 0)
    
    // Step 4: Normalize
    let normBufferSafe = MLX.maximum(normBuffer, 1e-10)
    let normalized = output / normBufferSafe.expandedDimensions(axis: 0)
    
    // Step 5: Final trimming
    var result = normalized
    if center {
        let startCut = nFFT / 2
        result = result[0..., startCut...]
    }
    
    if let audioLength = audioLength {
        result = result[0..., 0..<audioLength]
    }
    
    return result
}*/

/// STFT that matches torchlibrosa interface.
public class STFT: Module {
    let n_fft: Int
    let hop_length: Int
    let win_length: Int
    let center: Bool
    let pad_mode: String
    let window_func: MLXArray
    
    public init(n_fft: Int = 2048, hop_length: Int? = nil, win_length: Int? = nil,
                window: String = "hann", center: Bool = true, pad_mode: String = "reflect",
                freeze_parameters: Bool = true) {
        self.n_fft = n_fft
        self.hop_length = hop_length ?? n_fft / 4
        self.win_length = win_length ?? n_fft
        self.center = center
        self.pad_mode = pad_mode
        
        // Create window
        if window == "hann" {
            self.window_func = createHannWindow(self.win_length, periodic: true)
        } else {
            fatalError("Window \(window) not supported")
        }
        
        super.init()
    }
    
    private func _forwardImpl(_ x: MLXArray) -> (real: MLXArray, imag: MLXArray) {
        
        // Handle input shape
        var input = x
        if x.ndim == 2 {
            // (batch, samples) -> (batch, 1, samples)
            input = input.expandedDimensions(axis: 1)
        }
        
        let batch = input.shape[0]
        let channels = input.shape[1]
        let samples = input.shape[2]
        
        // Reshape to process all channels at once: (batch * channels, samples)
        let xFlat = input.reshaped([batch * channels, samples])
        
        // Apply STFT to all channels simultaneously
        let (real, imag) = stft(xFlat, nFFT: n_fft, hopLength: hop_length,
                                     winLength: win_length, window: window_func, center: center)
        
        // real, imag have shape (batch * channels, freq, time)
        // Reshape back to (batch, channels, freq, time)
        let freqBins = real.shape[1]
        let timeFrames = real.shape[2]
        
        var realReshaped = real.reshaped([batch, channels, freqBins, timeFrames])
        var imagReshaped = imag.reshaped([batch, channels, freqBins, timeFrames])
        
        // Transpose to (batch, channels, time, freq)
        realReshaped = realReshaped.transposed(0, 1, 3, 2)
        imagReshaped = imagReshaped.transposed(0, 1, 3, 2)
        
        return (realReshaped, imagReshaped)
    }
    
    public func callAsFunction(_ x: MLXArray) -> (real: MLXArray, imag: MLXArray) {
        /// Forward pass.
        ///
        /// - Parameter x: Input tensor of shape (batch, samples) or (batch, channels, samples)
        /// - Returns: Tuple containing real and imaginary parts of STFT, shape (batch, channels, time, freq)
        return _forwardImpl(x)
    }

    /// Number of frames produced by ``stft`` for this configuration.
    func frameCount(forSignalLength signalLength: Int) -> Int {
        guard signalLength > 0 else { return 0 }
        let padding = center ? 2 * (n_fft / 2) : 0
        let paddedLength = signalLength + padding
        guard paddedLength >= win_length else { return 0 }
        return (paddedLength - win_length) / hop_length + 1
    }
}

/// ISTFT that matches torchlibrosa interface.
public class ISTFT: Module {
    let n_fft: Int
    let hop_length: Int
    let win_length: Int
    let center: Bool
    let pad_mode: String
    let window_func: MLXArray
    
    public init(n_fft: Int = 2048, hop_length: Int? = nil, win_length: Int? = nil,
                window: String = "hann", center: Bool = true, pad_mode: String = "reflect",
                freeze_parameters: Bool = true) {
        self.n_fft = n_fft
        self.hop_length = hop_length ?? n_fft / 4
        self.win_length = win_length ?? n_fft
        self.center = center
        self.pad_mode = pad_mode
        
        // Create window
        if window == "hann" {
            self.window_func = createHannWindow(self.win_length, periodic: true)
        } else {
            fatalError("Window \(window) not supported")
        }
        
        super.init()
    }
    
    private func _forwardImpl(real: MLXArray, imag: MLXArray, length: Int?) -> MLXArray {
        let batch = real.shape[0]
        let channels = real.shape[1]
        let time = real.shape[2]
        let freq = real.shape[3]
        
        // Transpose to (batch, channels, freq, time) then reshape
        let realTransposed = real.transposed(0, 1, 3, 2)
        let imagTransposed = imag.transposed(0, 1, 3, 2)
        
        // Reshape to process all channels at once: (batch * channels, freq, time)
        let realFlat = realTransposed.reshaped([batch * channels, freq, time])
        let imagFlat = imagTransposed.reshaped([batch * channels, freq, time])
        
        // Apply ISTFT to all channels simultaneously
        let xFlat = istft(realPart: realFlat, imagPart: imagFlat, nFFT: n_fft, hopLength: hop_length,
                              winLength: win_length, window: window_func, center: center, audioLength: length)
        
        // x_flat has shape (batch * channels, samples)
        let samples = xFlat.shape[1]
        
        // Reshape back to (batch, channels, samples)
        var x = xFlat.reshaped([batch, channels, samples])
        
        // If single channel, squeeze channel dimension
        if channels == 1 {
            x = x[0..., 0, 0...]  // (batch, samples)
        }
        
        return x
    }
    
    public func callAsFunction(real: MLXArray, imag: MLXArray, length: Int? = nil) -> MLXArray {
        /// Forward pass.
        ///
        /// - Parameters:
        ///   - real: Real part of STFT, shape (batch, channels, time, freq)
        ///   - imag: Imaginary part of STFT, shape (batch, channels, time, freq)
        ///   - length: Original signal length
        /// - Returns: Reconstructed signal, shape (batch, samples) or (batch, channels, samples)
        return _forwardImpl(real: real, imag: imag, length: length)
    }

    /// Materialize the normalization entry that this ISTFT instance will use.
    /// Cache identity includes `window_func`, so prewarming must use this owned
    /// window rather than a newly-created window with equivalent values.
    func prewarmNormalization(numFrames: Int) {
        guard numFrames > 0, window_func.shape == [n_fft] else { return }
        _ = _mlxNormBufferCache.value(
            numFrames: numFrames,
            hopLength: hop_length,
            frameLength: n_fft,
            window: window_func
        ) {
            createISTFTNormBuffer(
                nFFT: n_fft,
                hopLength: hop_length,
                winLength: win_length,
                window: window_func,
                numFrames: numFrames
            )
        }
    }

    func isNormalizationPrewarmed(numFrames: Int) -> Bool {
        _mlxNormBufferCache.contains(
            numFrames: numFrames,
            hopLength: hop_length,
            frameLength: n_fft,
            window: window_func
        )
    }
}

// MARK: - MLX-Optimized iSTFT Implementation

private struct ISTFTNormCacheKey: Hashable, CustomStringConvertible {
    let numFrames: Int
    let hopLength: Int
    let frameLength: Int
    let windowID: ObjectIdentifier

    var description: String {
        "\(numFrames)_\(hopLength)_\(frameLength)_\(windowID)"
    }
}

/// A normalization buffer depends on the actual window, not just its shape.
/// Retaining the window alongside the buffer also prevents ObjectIdentifier
/// reuse while the cache entry exists.
private final class ISTFTNormBufferCache: @unchecked Sendable {
    private struct Entry {
        let window: MLXArray
        let buffer: MLXArray
    }

    private let lock = NSLock()
    private let maximumEntries = 32
    private var entries: [ISTFTNormCacheKey: Entry] = [:]

    func value(
        numFrames: Int,
        hopLength: Int,
        frameLength: Int,
        window: MLXArray,
        create: () -> MLXArray
    ) -> MLXArray {
        let key = ISTFTNormCacheKey(
            numFrames: numFrames,
            hopLength: hopLength,
            frameLength: frameLength,
            windowID: ObjectIdentifier(window)
        )

        lock.lock()
        if let cached = entries[key]?.buffer {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let candidate = create()
        eval(candidate)

        lock.lock()
        defer { lock.unlock() }
        if let cached = entries[key]?.buffer {
            return cached
        }
        if entries.count >= maximumEntries, let oldest = entries.keys.first {
            entries.removeValue(forKey: oldest)
        }
        entries[key] = Entry(window: window, buffer: candidate)
        return candidate
    }

    func contains(
        numFrames: Int,
        hopLength: Int,
        frameLength: Int,
        window: MLXArray
    ) -> Bool {
        let key = ISTFTNormCacheKey(
            numFrames: numFrames,
            hopLength: hopLength,
            frameLength: frameLength,
            windowID: ObjectIdentifier(window)
        )
        lock.lock()
        defer { lock.unlock() }
        return entries[key] != nil
    }
}

private let _mlxNormBufferCache = ISTFTNormBufferCache()

/// MLX-optimized iSTFT using scatter-add operations like Python
public func istft(
    realPart: MLXArray,
    imagPart: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true,
    audioLength: Int? = nil
) -> MLXArray {
    // Step 1: Get windowed time-domain frames
    let stftComplex = realPart + MLXArray(real: 0, imaginary: 1) * imagPart
    let timeFrames = MLX.irfft(stftComplex.transposed(0, 2, 1), n: nFFT, axis: -1)
    let windowedFrames = timeFrames * window
    
    let batchSize = windowedFrames.shape[0]
    let numFrames = windowedFrames.shape[1]
    let frameLength = windowedFrames.shape[2]
    let olaLen = (numFrames - 1) * hopLength + frameLength
    
    // Step 2: Pre-compute normalization buffer (with caching)
    let normBuffer = _mlxNormBufferCache.value(
        numFrames: numFrames,
        hopLength: hopLength,
        frameLength: frameLength,
        window: window
    ) {
        let windowSquared = window * window
        var buffer = MLXArray.zeros([olaLen], dtype: .float32)
        
        // Vectorized normalization buffer creation using MLX operations
        let positions = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                       MLXArray(0..<frameLength).expandedDimensions(axis: 0)
        let positionsFlat = positions.flattened()
        let windowSqTiled = tiled(windowSquared, repetitions: [numFrames])
        
        // Use MLX's at[].add() for efficient scatter-add
        buffer = buffer.at[positionsFlat].add(windowSqTiled)
        
        return MLX.maximum(buffer, MLXArray(Float(1e-8)))
    }
    
    // Step 3: Optimized overlap-add using MLX scatter operations
    var output = MLXArray.zeros([batchSize, olaLen], dtype: .float32)
    
    // Reshape for vectorized scatter
    let windowedFlat = windowedFrames.reshaped([batchSize, -1])
    
    // Pre-compute positions
    let positions = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                   MLXArray(0..<frameLength).expandedDimensions(axis: 0)
    let positionsFlat = positions.flattened()
    
    // Use scatter-add for all batches
    for b in 0..<batchSize {
        output[b] = output[b].at[positionsFlat].add(windowedFlat[b])
    }
    
    // Step 4: Apply normalization
    output = output / normBuffer.expandedDimensions(axis: 0)
    
    // Step 5: Apply center trimming
    if center {
        let startCut = nFFT / 2
        output = output[0..., startCut...]
    }
    
    // Step 6: Apply length adjustment
    if let audioLength = audioLength {
        let currentLen = output.shape[1]
        if currentLen > audioLength {
            output = output[0..., 0..<audioLength]
        } else if currentLen < audioLength {
            // Pad with zeros - this is the correct approach
            let padAmount = audioLength - currentLen
            let padding = MLXArray.zeros([output.shape[0], padAmount], dtype: output.dtype)
            output = MLX.concatenated([output, padding], axis: 1)
        }
    }
    
    return output
}

/// Create normalization buffer for caching
public func createISTFTNormBuffer(
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    numFrames: Int
) -> MLXArray {
    let frameLength = window.shape[0]
    let olaLen = (numFrames - 1) * hopLength + frameLength
    
    let windowSquared = window * window
    var normBuffer = MLXArray.zeros([olaLen], dtype: .float32)
    
    // Vectorized creation using MLX operations
    let positions = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                   MLXArray(0..<frameLength).expandedDimensions(axis: 0)
    let positionsFlat = positions.flattened()
    let windowSqTiled = tiled(windowSquared, repetitions: [numFrames])
    
    normBuffer = normBuffer.at[positionsFlat].add(windowSqTiled)
    
    return MLX.maximum(normBuffer, MLXArray(Float(1e-8)))
}

/// MLX-optimized iSTFT with pre-computed normalization buffer
/*public func mlxOptimizedISTFTCached(
    realPart: MLXArray,
    imagPart: MLXArray,
    nFFT: Int,
    hopLength: Int,
    normBuffer: MLXArray,
    window: MLXArray,
    center: Bool = true,
    audioLength: Int? = nil
) -> MLXArray {
    // Step 1: Get windowed time-domain frames
    let stftComplex = realPart + MLXArray(real: 0, imaginary: 1) * imagPart
    let timeFrames = irfft(stftComplex.transposed(0, 2, 1), n: nFFT, axis: -1)
    let windowedFrames = timeFrames * window
    
    let batchSize = windowedFrames.shape[0]
    let numFrames = windowedFrames.shape[1]
    let frameLength = windowedFrames.shape[2]
    let olaLen = normBuffer.shape[0]
    
    // Step 2: Pre-compute positions
    let positions = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                   MLXArray(0..<frameLength).expandedDimensions(axis: 0)
    let positionsFlat = positions.flattened()
    
    // Step 3: Fast overlap-add using scatter operations
    var output = MLXArray.zeros([batchSize, olaLen], dtype: .float32)
    let windowedFlat = windowedFrames.reshaped([batchSize, -1])
    
    for b in 0..<batchSize {
        output[b] = output[b].at[positionsFlat].add(windowedFlat[b])
    }
    
    // Step 4: Use pre-computed normalization
    output = output / normBuffer.expandedDimensions(axis: 0)
    
    // Step 5: Apply trimming and length adjustment
    if center {
        let startCut = nFFT / 2
        output = output[0..., startCut...]
    }
    
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
}*/

// MARK: - Cache Management

/// Clear MLX norm buffer cache
/*public func clearMLXNormBufferCache() {
    _mlxNormBufferCache.removeAll()
}*/

/// Get MLX cache information
/*public func getMLXCacheInfo() -> (items: Int, keys: [String]) {
    return (items: _mlxNormBufferCache.count, keys: Array(_mlxNormBufferCache.keys))
}*/

// MARK: Ultra-fast STFT

/// Ultra-fast STFT (STFTVariants) implementation based on Python MLX optimization patterns
public func stft(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true
) -> (MLXArray, MLXArray) {
    // Ensure 2D input
    let input2D = x.ndim == 1 ? x.expandedDimensions(axis: 0) : x
    let batchSize = input2D.shape[0]
    let signalLen = input2D.shape[1]
    
    // Ultra-efficient padding
    let paddedSignal: MLXArray
    let paddedLen: Int
    
    if center {
        let padAmount = nFFT / 2
        // Pre-compute slice ranges for efficiency
        let leftStart = 1
        let leftEnd = min(padAmount + 1, signalLen)
        let rightStart = max(0, signalLen - padAmount - 1)
        let rightEnd = signalLen - 1
        
        // Single concatenation with reversed slices
        paddedSignal = MLX.concatenated([
            input2D[0..., leftStart..<leftEnd][0..., .stride(by: -1)],
            input2D,
            input2D[0..., rightStart..<rightEnd][0..., .stride(by: -1)]
        ], axis: -1)
        paddedLen = signalLen + 2 * padAmount
    } else {
        paddedSignal = input2D
        paddedLen = signalLen
    }
    
    // Frame extraction with optimized strides
    let numFrames = (paddedLen - winLength) / hopLength + 1
    guard numFrames > 0 else {
        let freqBins = (nFFT / 2) + 1
        return (MLXArray.zeros([batchSize, freqBins, 1]),
                MLXArray.zeros([batchSize, freqBins, 1]))
    }
    
    // Optimized strided view
    let frames = MLX.asStrided(
        paddedSignal,
        [batchSize, numFrames, winLength],
        strides: [paddedLen, hopLength, 1]
    )
    
    // Window application and FFT preparation
    var windowedFrames = frames * window
    
    // Pad if necessary
    if winLength < nFFT {
        let zeros = MLXArray.zeros([batchSize, numFrames, nFFT - winLength])
        windowedFrames = MLX.concatenated([windowedFrames, zeros], axis: -1)
    }
    
    // Single FFT call with immediate transpose
    let spectrum = MLX.rfft(windowedFrames, n: nFFT, axis: -1).transposed(0, 2, 1)
    
    return (spectrum.realPart(), spectrum.imaginaryPart())
}
