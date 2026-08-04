import MLX
import MLXNN

// MARK: - Utility Functions

/// Pad a tensor along a specific axis with reflect or constant padding
/// - Parameters:
///   - x: Input array
///   - paddings: Tuple of (left, right) padding amounts
///   - mode: "reflect" or "constant"
///   - value: Constant value for padding (only used in constant mode)
///   - axis: Axis to pad along
/// - Returns: Padded array
public func pad1d(
    _ x: MLXArray,
    paddings: (Int, Int),
    mode: String = "constant",
    value: Float = 0.0,
    axis: Int = 1
) -> MLXArray {
    let (padLeft, padRight) = paddings
    let ndim = x.ndim
    
    // Normalize negative axis
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    
    if mode == "reflect" {
        var parts: [MLXArray] = []
        
        // For reflection we need a simpler approach using transpose
        // Move the target axis to the last position, slice, reverse, move back
        
        // Reflect left
        if padLeft > 0 {
            // x[1:padLeft+1] along axis, then reversed
            // Use take for indexed access
            let indices = MLXArray((1..<(padLeft + 1)).reversed().map { Int32($0) })
            let leftPart = x.take(indices, axis: normalizedAxis)
            parts.append(leftPart)
        }
        
        parts.append(x)
        
        // Reflect right
        if padRight > 0 {
            // x[-(padRight+1):-1] along axis, then reversed
            let axisLen = x.shape[normalizedAxis]
            let start = axisLen - padRight - 1
            let end = axisLen - 1
            let indices = MLXArray((start..<end).reversed().map { Int32($0) })
            let rightPart = x.take(indices, axis: normalizedAxis)
            parts.append(rightPart)
        }
        
        return MLX.concatenated(parts, axis: normalizedAxis)
        
    } else if mode == "constant" {
        // Build pad width array for mx.pad
        var widths: [IntOrPair] = []
        for i in 0..<ndim {
            if i == normalizedAxis {
                widths.append(IntOrPair([padLeft, padRight]))
            } else {
                widths.append(IntOrPair(0))
            }
        }
        return MLX.padded(x, widths: widths, value: MLXArray(value))
        
    } else {
        fatalError("Unsupported padding mode: \(mode)")
    }
}

/// Trim a tensor to a given length along a specific axis (center crop)
/// - Parameters:
///   - x: Input array
///   - length: Target length
///   - axis: Axis to trim along
/// - Returns: Trimmed array
public func centerTrim(_ x: MLXArray, length: Int, axis: Int = 1) -> MLXArray {
    let ndim = x.ndim
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    let currentLength = x.shape[normalizedAxis]
    
    guard currentLength >= length else {
        fatalError("Input length \(currentLength) along axis \(normalizedAxis) is smaller than target length \(length)")
    }
    
    let delta = currentLength - length
    let start = delta / 2
    
    // Use take with sequential indices for center trim
    let indices = MLXArray((start..<(start + length)).map { Int32($0) })
    return x.take(indices, axis: normalizedAxis)
}
