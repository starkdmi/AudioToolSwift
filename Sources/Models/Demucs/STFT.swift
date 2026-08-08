import Foundation
import MLX
import MLXNN

// MARK: - Array Striding Utilities
// Using MLX native asStrided for optimal performance

// MARK: - Window Creation Functions

private struct PeriodicHannWindowKey: Hashable {
    let length: Int
    let dtype: DType
}

/// Demucs constructs its Hann window at every spectro/ispectro call. Returning a
/// stable immutable instance avoids rebuilding the window and lets the iSTFT
/// normalization cache use object identity without conflating different content.
private final class PeriodicHannWindowCache: @unchecked Sendable {
    private let lock = NSLock()
    private let maximumEntries = 16
    private var entries: [PeriodicHannWindowKey: MLXArray] = [:]
    private var insertionOrder: [PeriodicHannWindowKey] = []

    func value(
        for key: PeriodicHannWindowKey,
        create: () -> MLXArray
    ) -> MLXArray {
        lock.lock()
        if let cached = entries[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let candidate = create()
        eval(candidate)

        lock.lock()
        defer { lock.unlock() }
        if let cached = entries[key] {
            return cached
        }
        if entries.count >= maximumEntries, let oldest = insertionOrder.first {
            entries.removeValue(forKey: oldest)
            insertionOrder.removeFirst()
        }
        entries[key] = candidate
        insertionOrder.append(key)
        return candidate
    }
}

private let periodicHannWindowCache = PeriodicHannWindowCache()

public func createPeriodicHannWindow(winLength: Int, dtype: DType = .float32) -> MLXArray {
    periodicHannWindowCache.value(
        for: PeriodicHannWindowKey(length: winLength, dtype: dtype)
    ) {
        // Mathematically identical to torch.hann_window(periodic=True).
        let n = MLXArray(0..<winLength).asType(dtype)
        let pi = dtype == .float16 ? MLXArray(Float.pi).asType(.float16) : MLXArray(Float.pi)
        let length = dtype == .float16 ? MLXArray(Float(winLength)).asType(.float16) : MLXArray(Float(winLength))
        let half = dtype == .float16 ? MLXArray(Float(0.5)).asType(.float16) : MLXArray(Float(0.5))
        let one = dtype == .float16 ? MLXArray(Float(1.0)).asType(.float16) : MLXArray(Float(1.0))
        let two = dtype == .float16 ? MLXArray(Float(2.0)).asType(.float16) : MLXArray(Float(2.0))

        return half * (one - MLX.cos(two * pi * n / length))
    }
}

// MARK: - Fast STFT Implementation (3.71x speedup over previous version)

/// High-performance STFT implementation using native MLX Swift operations
/// - Parameters:
///   - x: Input signal [batch, samples] or [samples]
///   - nFFT: FFT size
///   - hopLength: Hop length between frames
///   - winLength: Window length
///   - window: Window function
///   - center: Whether to center the signal
/// - Returns: Tuple of (real, imaginary) parts with shape [batch, freq, time]
public func stft(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true
) -> (MLXArray, MLXArray) {
    let shape = x.shape
    
    // Validate input
    guard shape.count >= 1 && (shape.count == 1 || shape[0] > 0) else {
        let emptyResult = MLXArray.zeros([1, (nFFT / 2) + 1, 1], dtype: x.dtype)
        return (emptyResult, emptyResult)
    }
    
    // Handle both 1D and 2D inputs
    let batchSize: Int
    let inputFor2D: MLXArray
    
    if shape.count == 1 {
        // 1D input: convert to 2D
        batchSize = 1
        inputFor2D = x.expandedDimensions(axis: 0)
    } else {
        // 2D input
        batchSize = shape[0]
        inputFor2D = x
    }
    
    // Process all batches at once using vectorized operations
    var paddedSignals = inputFor2D
    
    if center {
        // Vectorized padding for all batches
        let padAmount = nFFT / 2
        
        // Use manual reflection padding since MLX Swift doesn't have reflect mode
        let leftPad = paddedSignals[0..., 1..<(padAmount + 1)][0..., .stride(by: -1)]
        let rightPad = paddedSignals[0..., -(padAmount + 1)..<(-1)][0..., .stride(by: -1)]
        paddedSignals = MLX.concatenated([leftPad, paddedSignals, rightPad], axis: 1)
    }
    
    // Calculate number of frames
    let signalLength = paddedSignals.shape[1]
    let numFrames = (signalLength - winLength) / hopLength + 1
    
    if numFrames <= 0 {
        let emptyResult = MLXArray.zeros([batchSize, (nFFT / 2) + 1, 1], dtype: x.dtype)
        return (emptyResult, emptyResult)
    }
    
    // Pad window to nFFT size first (matching Python behavior)
    var paddedWindow = window
    if winLength < nFFT {
        let padAmount = nFFT - winLength
        paddedWindow = MLX.concatenated([window, MLXArray.zeros([padAmount], dtype: window.dtype)], axis: 0)
    }
    
    // Pad signal if needed to ensure we have enough samples for nFFT-sized frames
    let targetLen = (numFrames - 1) * hopLength + nFFT
    var signalForStrided = paddedSignals
    if signalLength < targetLen {
        let padExtra = targetLen - signalLength
        signalForStrided = MLX.concatenated([
            paddedSignals,
            MLXArray.zeros([batchSize, padExtra], dtype: paddedSignals.dtype)
        ], axis: 1)
    }
    
    // Use native asStrided for efficient frame extraction
    // Create frames of size nFFT (not winLength) - matching Python behavior
    let frames = MLX.asStrided(
        signalForStrided,
        [batchSize, numFrames, nFFT],
        strides: [signalForStrided.shape[1], hopLength, 1]
    )
    
    // Apply padded window to all frames at once
    let windowedFrames = frames * paddedWindow
    
    // Compute FFT for all frames at once
    let stftComplex = rfft(windowedFrames, n: nFFT, axis: -1)
    
    // Transpose to match expected output shape [batch, freq, time]
    let stftTransposed = stftComplex.transposed(0, 2, 1)
    
    let realPart = stftTransposed.realPart()
    let imagPart = stftTransposed.imaginaryPart()
    
    return (realPart, imagPart)
}

// MARK: - Old STFT Implementation (Commented Out)

/*
// Kokoro STFT function adapted from their implementation
func kokoroSTFT(
    x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true,
    padMode: String = "reflect"
) -> MLXArray {
    // Debug logging for streaming issue
    if x.shape.isEmpty || (x.shape.count > 0 && x.shape[0] == 0) {
        print("⚠️ kokoroSTFT received empty or invalid input: shape=\(x.shape)")
    }
    
    var w = window
    
    // Ensure window length matches nFFT
    if w.shape[0] < nFFT {
        let padSize = nFFT - w.shape[0]
        let zeros = MLXArray.zeros([padSize], dtype: w.dtype)
        w = MLX.concatenated([w, zeros], axis: 0)
    }
    
    var xArray = x
    
    if center {
        xArray = padSignal(xArray, padding: nFFT / 2, padMode: padMode)
    }
    
    let numFrames = 1 + (xArray.shape[0] - nFFT) / hopLength
    
    if numFrames <= 0 {
        print("⚠️ kokoroSTFT: Input too short for STFT - numFrames=\(numFrames). Returning empty result.")
        return Float16Utils.optimizedZeros([(nFFT / 2) + 1, 1], referenceDType: x.dtype)
    }
    
    let shape: [Int] = [numFrames, nFFT]
    let strides: [Int] = [hopLength, 1]
    
    let frames = MLX.asStrided(xArray, shape, strides: strides)
    
    let windowedFrames = frames * w
    let spec = rfft(windowedFrames, n: nFFT, axis: -1)
    
    // Note: MLX Swift only has complex64, so we'll extract and convert components later
    
    return spec.transposed(1, 0)  // [freq, time]
}

// Kokoro padding function
// Note: This function only handles 1D signals. Batch processing is handled by the calling function.
func padSignal(_ x: MLXArray, padding: Int, padMode: String = "reflect") -> MLXArray {
    // Validate input
    guard x.shape.count == 1 else {
        print("⚠️ padSignal: Expected 1D array, got shape: \(x.shape). Using fallback zero padding.")
        // Fallback: apply zero padding for unexpected input shapes
        if x.shape.count == 2 {
            return MLX.padded(x, widths: [IntOrPair(0), IntOrPair(padding)])
        } else {
            return MLX.padded(x, widths: [IntOrPair(padding)])
        }
    }
    
    let signalLength = x.shape[0]
    
    // Handle empty signals
    guard signalLength > 0 else {
        print("⚠️ padSignal: Empty signal, returning zero padding")
        return Float16Utils.createPadding(shape: [2 * padding], referenceDType: x.dtype)
    }
    
    if padMode == "constant" {
        return MLX.padded(x, widths: [IntOrPair(padding)])
    } else if padMode == "reflect" {
        // Handle edge case where signal is too short for full reflection padding
        if signalLength == 1 {
            // Special case: single sample - replicate it
            let singleValue = x[0].item() as Float
            let padArray = MLXArray(Array(repeating: singleValue, count: padding)).asType(x.dtype)
            return MLX.concatenated([padArray, x, padArray])
        } else if signalLength <= padding {
            // Short signal: use cyclic reflection
            var prefix = [Float]()
            var suffix = [Float]()
            
            // Build prefix by cycling through available samples
            var idx = 1
            for _ in 0..<padding {
                prefix.append(x[idx].item() as Float)
                idx = (idx + 1) % signalLength
                if idx == 0 { idx = 1 }  // Skip the first sample to maintain reflection
            }
            prefix.reverse()
            
            // Build suffix similarly
            idx = signalLength - 2
            for _ in 0..<padding {
                suffix.append(x[idx].item() as Float)
                idx = idx - 1
                if idx < 0 { idx = signalLength - 2 }  // Wrap around, skip the last sample
            }
            
            let prefixArray = MLXArray(prefix).asType(x.dtype)
            let suffixArray = MLXArray(suffix).asType(x.dtype)
            
            return MLX.concatenated([prefixArray, x, suffixArray])
        } else {
            // Normal case: signal is long enough for standard reflection padding
            // Use MLX native asStrided for efficient negative stride operations
            let prefixRange = 1..<(padding + 1)
            let suffixStart = signalLength - padding - 1
            let suffixEnd = signalLength - 1
            let suffixRange = suffixStart..<suffixEnd
            
            let prefix = x[prefixRange][.stride(by: -1)]
            let suffix = x[suffixRange][.stride(by: -1)]
            return MLX.concatenated([prefix, x, suffix])
        }
    } else {
        print("⚠️ padSignal: Invalid pad mode '\(padMode)', using constant padding")
        return MLX.padded(x, widths: [IntOrPair(padding)])
    }
}
*/

// MARK: - iSTFT Implementation

// https://github.com/iliasaz/kokoro-swift
public func istft_1( // MARK: REPLACED BY mlxOptimizedISTFT)
    realPart: MLXArray,
    imagPart: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true,
    audioLength: Int? = nil
) -> MLXArray {
    // Removed debug output for performance
    // Use Kokoro-style iSTFT implementation for better accuracy
    
    // Step 1: Create complex STFT array from real/imag components
    // stft_complex = real_part + 1j * imag_part
    // Create complex array (always complex64 in MLX Swift)
    let stftComplex = realPart + MLXArray(real: 0, imaginary: 1) * imagPart
    
    // Transpose from [batch, freq, time] to [batch, time, freq] to match Python
    let stftTransposed = stftComplex.transposed(0, 2, 1)
    
    let batchSize = stftTransposed.shape[0]
    let numFrames = stftTransposed.shape[1]
    let _ = stftTransposed.shape[2]  // freqBins - not used
    
    let expectedLength = (numFrames - 1) * hopLength + winLength
    
    var batchResults: [MLXArray] = []
    
    // Process each batch item separately (like Kokoro does)
    for b in 0..<batchSize {
        // Apply irfft to get time-domain frames: [time, freq] -> [time, win_length]
        var timeFrames = irfft(stftTransposed[b], n: nFFT, axis: 1)
        
        // Determine target dtype from input - prefer Float16 for memory efficiency
        let targetDtype = realPart.dtype
        
        // Force conversion to target dtype if needed for consistency
        if timeFrames.dtype != targetDtype {
            timeFrames = timeFrames.asType(targetDtype)
        }
        
        // Apply window - ensure window matches target dtype
        let adjustedWindow = window.dtype == targetDtype ? window : window.asType(targetDtype)
        let windowedFrames = timeFrames * adjustedWindow
        
        // Overlap-add reconstruction - optimized with MLX vectorized operations
        var output = Float16Utils.createIntermediateTensor(shape: [expectedLength], dtype: targetDtype)
        var normBuffer = Float16Utils.createIntermediateTensor(shape: [expectedLength], dtype: targetDtype)
        
        // Optimized overlap-add using MLX.scatter operations for maximum performance
        let windowSquared = adjustedWindow * adjustedWindow
        
        // Pre-compute all frame positions for vectorized scatter operations
        let frameIdxArray = MLXArray(0..<numFrames) // [numFrames]
        let hopArray = MLXArray(Array(repeating: hopLength, count: numFrames)) // [numFrames]
        let frameStartPositions = frameIdxArray * hopArray // [numFrames]
        
        // Create index arrays for scatter operations using MLX.asStrided
        let winIdxArray = MLXArray(0..<winLength) // [winLength]
        
        // Use broadcasting to create all position indices at once
        // frameStartPositions: [numFrames, 1], winIdxArray: [1, winLength]
        let expandedStarts = frameStartPositions.expandedDimensions(axis: 1) // [numFrames, 1]
        let expandedWinIdx = winIdxArray.expandedDimensions(axis: 0) // [1, winLength]
        let allPositions = expandedStarts + expandedWinIdx // [numFrames, winLength]
        
        // Flatten for scatter operations
        let flatPositions = allPositions.flattened() // [numFrames * winLength]
        let flatFrameData = windowedFrames.flattened() // [numFrames * winLength]
        let flatWindowData = MLX.stacked(Array(repeating: windowSquared, count: numFrames), axis: 0).flattened()
        
        // Handle boundary conditions - clip positions to valid range
        let maxValidPos = MLXArray(expectedLength - 1)
        let clippedPositions = MLX.clip(flatPositions, min: MLXArray(0), max: maxValidPos)
        
        // Use MLX scatter-add for efficient overlap-add reconstruction
        output = output.at[clippedPositions].add(flatFrameData)
        normBuffer = normBuffer.at[clippedPositions].add(flatWindowData)
        
        // Normalize to avoid division by zero - use appropriate epsilon for dtype
        let epsilon = targetDtype == .float16 ? MLXArray(Float(1e-7)).asType(.float16) : MLXArray(Float(1e-10))
        normBuffer = MLX.maximum(normBuffer, epsilon)
        output = output / normBuffer
        
        batchResults.append(output)
    }
    
    // Stack batch results
    var result = MLX.stacked(batchResults, axis: 0)
    
    // Apply center trimming if needed
    if center {
        let trimAmount = nFFT / 2
        let resultLen = result.shape[1]
        if resultLen > 2 * trimAmount {
            result = result[0..., trimAmount..<(resultLen - trimAmount)]
        }
    }
    
    // Apply length trimming if specified
    if let audioLength = audioLength {
        let currentLen = result.shape[1]
        if currentLen > audioLength {
            result = result[0..., 0..<audioLength]
        }
    }
    
    return result
}

// MARK: Cached STFT

public struct STFTPaddingIndices {
    let leftIndices: MLXArray
    let rightIndices: MLXArray
    let paddedLength: Int
}

public func createSTFTPaddingIndices(signalLength: Int, nFFT: Int, center: Bool = true) -> STFTPaddingIndices? {
    guard center else { return nil }
    
    let padAmount = nFFT / 2
    
    // Handle edge case where signal is too short
    let leftIndices: MLXArray
    let rightIndices: MLXArray
    
    if signalLength <= padAmount {
        // For very short signals, use available samples
        leftIndices = MLXArray(Array(1..<min(signalLength, padAmount + 1)))
        rightIndices = MLXArray(Array(0..<max(0, signalLength - 1)))
    } else {
        // Normal case
        leftIndices = MLXArray(Array(1..<(padAmount + 1)))
        let rightStart = max(0, signalLength - padAmount - 1)
        let rightEnd = signalLength - 1
        rightIndices = MLXArray(Array(rightStart..<rightEnd))
    }
    
    let paddedLength = signalLength + 2 * padAmount
    
    return STFTPaddingIndices(
        leftIndices: leftIndices,
        rightIndices: rightIndices,
        paddedLength: paddedLength
    )
}

public func cachedSTFT(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    paddingIndices: STFTPaddingIndices? = nil,
    center: Bool = true
) -> (MLXArray, MLXArray) {
    // print("🔧 cachedSTFT: input=\(x.shape), nFFT=\(nFFT), hopLength=\(hopLength), winLength=\(winLength)")
    
    let _ = x.dtype == .float16  // Track Float16 usage for potential optimization
    
    // Handle both 1D and 2D input
    let batchSize: Int
    let signalLength: Int
    
    if x.shape.count == 1 {
        batchSize = 1
        signalLength = x.shape[0]
    } else {
        batchSize = x.shape[0]
        signalLength = x.shape[1]
    }
    
    // print("🔧 cachedSTFT: batchSize=\(batchSize), signalLength=\(signalLength)")
    
    // Ensure we have a 2D array for consistent processing
    var paddedX = x
    if x.shape.count == 1 {
        paddedX = x.expandedDimensions(axis: 0) // [L] -> [1, L]
        // print("🔧 cachedSTFT: Converted 1D input to 2D: \(x.shape) -> \(paddedX.shape)")
    }
    
    // Apply cached padding if available
    if center {
        if let indices = paddingIndices {
            // Validate indices before using them
            if indices.leftIndices.shape[0] > 0 && indices.rightIndices.shape[0] > 0 {
                // Use MLX native operations for efficient strided access
                let leftPad = paddedX[0..., indices.leftIndices][0..., .stride(by: -1)]
                let rightPad = paddedX[0..., indices.rightIndices][0..., .stride(by: -1)]
                paddedX = MLX.concatenated([leftPad, paddedX, rightPad], axis: -1)
            } else {
                // Fall back to zero padding if indices are invalid
                let padAmount = nFFT / 2
                let leftZeros = Float16Utils.createPadding(shape: [batchSize, padAmount], referenceDType: x.dtype)
                let rightZeros = Float16Utils.createPadding(shape: [batchSize, padAmount], referenceDType: x.dtype)
                paddedX = MLX.concatenated([leftZeros, paddedX, rightZeros], axis: -1)
            }
        } else {
            // Fallback padding
            let padAmount = nFFT / 2
            
            // Handle edge case where signal is too short
            if signalLength <= padAmount {
                // For very short signals, pad with zeros
                let leftZeros = Float16Utils.createPadding(shape: [batchSize, padAmount], referenceDType: x.dtype)
                let rightZeros = Float16Utils.createPadding(shape: [batchSize, padAmount], referenceDType: x.dtype)
                paddedX = MLX.concatenated([leftZeros, paddedX, rightZeros], axis: -1)
            } else {
                // Normal reflection padding
                let leftIndices = MLXArray(Array(1..<min(signalLength, padAmount + 1)))
                let rightStart = max(0, signalLength - padAmount - 1)
                let rightEnd = signalLength - 1
                let rightIndices = MLXArray(Array(rightStart..<rightEnd))
                
                // Use MLX native operations for efficient strided access
                let leftPad = paddedX[0..., leftIndices][0..., .stride(by: -1)]
                let rightPad = paddedX[0..., rightIndices][0..., .stride(by: -1)]
                
                // Pad with zeros if needed
                if leftPad.shape[1] < padAmount {
                    let zeros = Float16Utils.createPadding(shape: [batchSize, padAmount - leftPad.shape[1]], referenceDType: x.dtype)
                    let paddedLeft = MLX.concatenated([zeros, leftPad], axis: -1)
                    paddedX = MLX.concatenated([paddedLeft, paddedX, rightPad], axis: -1)
                } else {
                    paddedX = MLX.concatenated([leftPad, paddedX, rightPad], axis: -1)
                }
            }
        }
    }
    
    let paddedLength = paddedX.shape[1]
    let numFrames = (paddedLength - winLength) / hopLength + 1
    
    // print("🔧 cachedSTFT: paddedX.shape=\(paddedX.shape), paddedLength=\(paddedLength), numFrames=\(numFrames)")
    
    // Validate numFrames
    if numFrames <= 0 {
        print("⚠️ cachedSTFT: Invalid numFrames=\(numFrames). Returning empty result.")
        let emptyResult = Float16Utils.optimizedZeros([batchSize, (nFFT / 2) + 1, 1], referenceDType: x.dtype)
        return (emptyResult, emptyResult)
    }
    
    // Fast framing using as_strided equivalent
    let frameShape = [batchSize, numFrames, winLength]
    let frameStrides = [paddedLength, hopLength, 1]
    
    let frames = MLX.asStrided(paddedX, frameShape, strides: frameStrides)
    
    // Apply window
    var adjustedWindow = window
    if window.dtype != x.dtype {
        adjustedWindow = window.asType(x.dtype)
    }
    
    let windowedFrames = frames * adjustedWindow
    
    // Pad for FFT if needed
    var fftFrames = windowedFrames
    if winLength < nFFT {
        let padWidth = nFFT - winLength
        let zeros = Float16Utils.createPadding(shape: [batchSize, numFrames, padWidth], referenceDType: x.dtype)
        fftFrames = MLX.concatenated([windowedFrames, zeros], axis: -1)
    }
    
    // FFT and transpose
    let stftComplex = rfft(fftFrames, n: nFFT, axis: -1)
    let stftTransposed = stftComplex.transposed(0, 2, 1)
    
    let realPart = stftTransposed.realPart()
    let imagPart = stftTransposed.imaginaryPart()
    
    // Optimize cached STFT results for memory efficiency  
    let optimizedReal = Float16Utils.optimizeForSpectralProcessing(realPart)
    let optimizedImag = Float16Utils.optimizeForSpectralProcessing(imagPart)
    
    return (optimizedReal, optimizedImag)
}

public class STFTCache {
    private var paddingCache: [String: STFTPaddingIndices] = [:]
    private var windowCache: [String: MLXArray] = [:]
    private var cacheAccessCount: Int = 0
    private let maxCacheSize: Int
    private let clearAfterAccesses: Int
    
    public init(maxCacheSize: Int = 10, clearAfterAccesses: Int = 100) {
        self.maxCacheSize = maxCacheSize
        self.clearAfterAccesses = clearAfterAccesses
    }
    
    public func getPaddingIndices(signalLength: Int, nFFT: Int, center: Bool = true) -> STFTPaddingIndices? {
        let key = "\(signalLength)_\(nFFT)_\(center)"
        
        // Check for periodic cleanup
        checkAndCleanCache()
        
        if let cached = paddingCache[key] {
            return cached
        }
        
        // Enforce cache size limit
        if paddingCache.count >= maxCacheSize {
            // Remove oldest entry (simple FIFO for now)
            if let firstKey = paddingCache.keys.first {
                paddingCache.removeValue(forKey: firstKey)
            }
        }
        
        let indices = createSTFTPaddingIndices(signalLength: signalLength, nFFT: nFFT, center: center)
        paddingCache[key] = indices
        return indices
    }
    
    public func getPaddedWindow(_ window: MLXArray, nFFT: Int) -> MLXArray {
        let winLength = window.shape[0]
        let key = "\(winLength)_\(nFFT)_\(window.dtype)"
        
        // Check for periodic cleanup
        checkAndCleanCache()
        
        if let cached = windowCache[key] {
            return cached
        }
        
        // Enforce cache size limit
        if windowCache.count >= maxCacheSize {
            // Remove oldest entry (simple FIFO for now)
            if let firstKey = windowCache.keys.first {
                windowCache.removeValue(forKey: firstKey)
            }
        }
        
        var paddedWindow = window
        if winLength < nFFT {
            let padAmount = nFFT - winLength
            let zeros = MLXArray.zeros([padAmount], dtype: window.dtype)
            paddedWindow = MLX.concatenated([window, zeros], axis: 0)
        }
        
        windowCache[key] = paddedWindow
        return paddedWindow
    }
    
    /// Clear all cached data to free memory
    public func clearCache() {
        paddingCache.removeAll()
        windowCache.removeAll()
        cacheAccessCount = 0
        
        // Force MLX memory cleanup
        MLX.eval([])
        GPU.clearCache()
    }
    
    /// Check if cache should be cleared based on access count or memory pressure
    private func checkAndCleanCache() {
        cacheAccessCount += 1
        
        // Periodic cleanup based on access count
        if cacheAccessCount >= clearAfterAccesses {
            clearCache()
            return
        }
        
        // Memory pressure-based cleanup
        // Check if we're under memory pressure
        let snapshot = GPU.snapshot()
        let totalGPUMemory = snapshot.activeMemory + snapshot.cacheMemory
        
        // Clear cache if GPU memory usage is high (>2GB)
        if totalGPUMemory > 2 * 1024 * 1024 * 1024 {
            clearCache()
        }
    }
    
    /// Get cache statistics
    public func cacheInfo() -> (paddingItems: Int, windowItems: Int, totalItems: Int) {
        let padding = paddingCache.count
        let window = windowCache.count
        return (padding, window, padding + window)
    }
    
    public func stft(
        _ x: MLXArray,
        nFFT: Int,
        hopLength: Int,
        winLength: Int,
        window: MLXArray,
        center: Bool = true
    ) -> (MLXArray, MLXArray) {
        let signalLength = x.shape.count == 1 ? x.shape[0] : x.shape[1]
        let paddingIndices = getPaddingIndices(signalLength: signalLength, nFFT: nFFT, center: center)
        
        return cachedSTFT(
            x,
            nFFT: nFFT,
            hopLength: hopLength,
            winLength: winLength,
            window: window,
            paddingIndices: paddingIndices,
            center: center
        )
    }
}

// ISTFTCache removed to avoid memory issues with MLX Metal
// The main performance benefit comes from cached STFT operations
// For iSTFT, use the original implementation which is already optimized

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

    func removeAll() {
        lock.lock()
        entries.removeAll()
        lock.unlock()
    }

    func info() -> (items: Int, keys: [String]) {
        lock.lock()
        defer { lock.unlock() }
        return (entries.count, entries.keys.map(\.description))
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
    let timeFrames = irfft(stftComplex.transposed(0, 2, 1), n: nFFT, axis: -1)
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
public func mlxOptimizedISTFTCached(
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
}

// MARK: - Cache Management

/// Clear MLX norm buffer cache
public func clearMLXNormBufferCache() {
    _mlxNormBufferCache.removeAll()
}

/// Get MLX cache information
public func getMLXCacheInfo() -> (items: Int, keys: [String]) {
    _mlxNormBufferCache.info()
}
