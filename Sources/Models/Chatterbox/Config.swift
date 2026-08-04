import Foundation

public struct T3Config {
    public var text_tokens_dict_size: Int = 704
    public var start_text_token: Int = 255
    public var stop_text_token: Int = 0
    public var max_text_tokens: Int = 2048

    public var speech_tokens_dict_size: Int = 8194
    public var start_speech_token: Int = 6561
    public var stop_speech_token: Int = 6562
    public var max_speech_tokens: Int = 4096

    public var llama_config_name: String = "Llama_520M"
    public var input_pos_emb: String = "learned"
    public var speech_cond_prompt_len: Int = 150

    public var encoder_type: String = "voice_encoder"
    public var speaker_embed_size: Int = 256
    public var use_perceiver_resampler: Bool = true
    public var emotion_adv: Bool = true

    public var n_channels: Int {
        LLAMA_CONFIGS[llama_config_name]?.hidden_size ?? 1024
    }

    public var is_multilingual: Bool {
        text_tokens_dict_size == 2454
    }

    public static func english_only() -> T3Config {
        var cfg = T3Config()
        cfg.text_tokens_dict_size = 704
        return cfg
    }

    public static func multilingual() -> T3Config {
        var cfg = T3Config()
        cfg.text_tokens_dict_size = 2454
        return cfg
    }
}

public struct ModelConfig {
    public var model_type: String = "chatterbox"
    public var t3_config: T3Config = .english_only()

    public var s3_sr: Int = 16000
    public var s3gen_sr: Int = 24000
    public var sample_rate: Int = 24000

    public var enc_cond_len: Int = 6 * 16000
    public var dec_cond_len: Int = 10 * 24000

    public var model_path: String? = nil

    public init() {}

    public init(t3_config: T3Config) {
        self.t3_config = t3_config
    }
}

public struct VoiceEncConfig {
    public var num_mels: Int = 40
    public var sample_rate: Int = 16000
    public var speaker_embed_size: Int = 256
    public var ve_hidden_size: Int = 256
    public var n_fft: Int = 400
    public var hop_size: Int = 160
    public var win_size: Int = 400
    public var fmax: Int = 8000
    public var fmin: Int = 0
    public var preemphasis: Float = 0.0
    public var mel_power: Float = 2.0
    public var mel_type: String = "amp"
    public var normalized_mels: Bool = false
    public var ve_partial_frames: Int = 160
    public var ve_final_relu: Bool = true
    public var stft_magnitude_min: Float = 1e-4

    public init() {}
}

public struct LlamaConfig {
    public var model_type: String
    public var vocab_size: Int
    public var hidden_size: Int
    public var num_hidden_layers: Int
    public var intermediate_size: Int
    public var num_attention_heads: Int
    public var num_key_value_heads: Int
    public var head_dim: Int
    public var max_position_embeddings: Int
    public var rms_norm_eps: Float
    public var rope_theta: Float
    public var rope_traditional: Bool
    public var attention_bias: Bool
    public var mlp_bias: Bool
    public var tie_word_embeddings: Bool
}

public let LLAMA_CONFIGS: [String: LlamaConfig] = [
    "Llama_520M": LlamaConfig(
        model_type: "llama",
        vocab_size: 8,
        hidden_size: 1024,
        num_hidden_layers: 30,
        intermediate_size: 4096,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        head_dim: 64,
        max_position_embeddings: 131072,
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        rope_traditional: true,
        attention_bias: false,
        mlp_bias: false,
        tie_word_embeddings: false
    )
]
