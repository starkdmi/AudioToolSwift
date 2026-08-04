import Foundation
import MLX

public struct GenerationResult {
    public var audio: MLXArray
    public var sample_rate: Int = 24000
    public var samples: Int = 0
    public var segment_idx: Int = 0
    public var token_count: Int = 0
    public var audio_duration: String = "00:00:00.000"
    public var real_time_factor: Float = 0.0
    public var prompt: [String: Any]? = nil
    public var audio_samples: [String: Any]? = nil
    public var processing_time_seconds: Double = 0.0
    public var peak_memory_usage: Float = 0.0
    public var tokens: MLXArray? = nil
    public var text: String? = nil
    public var duration: Double? = nil

    public init(audio: MLXArray) {
        self.audio = audio
    }
}
