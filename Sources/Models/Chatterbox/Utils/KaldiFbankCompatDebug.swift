#if DEBUG
import MLX

public enum KaldiFbankCompatDebug {
    public static func compute(_ audioIn: MLXArray) -> MLXArray {
        KaldiFbankCompat.compute(audioIn)
    }
}
#endif
