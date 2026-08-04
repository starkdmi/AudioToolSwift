import MLX

@inline(__always)
func expandDims(_ x: MLXArray, axis: Int) -> MLXArray {
    x.expandedDimensions(axis: axis)
}

@inline(__always)
func broadcastTo(_ x: MLXArray, _ shape: [Int]) -> MLXArray {
    broadcast(x, to: shape)
}

@inline(__always)
func zerosLike(_ x: MLXArray) -> MLXArray {
    MLX.zeros(like: x)
}

@inline(__always)
func stack(_ arrays: [MLXArray], axis: Int) -> MLXArray {
    precondition(!arrays.isEmpty)
    let rank = arrays[0].ndim
    let axisPos = axis < 0 ? axis + rank + 1 : axis
    let expanded = arrays.map { $0.expandedDimensions(axis: axisPos) }
    return MLX.concatenated(expanded, axis: axisPos)
}
