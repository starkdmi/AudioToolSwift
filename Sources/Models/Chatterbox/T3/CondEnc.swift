import MLX
import MLXNN

/// Conditioning for T3.
///
/// Upstream's struct also carries a `clap_emb`, and its conditioning encoder has
/// no implementation for one - the Python asserts CLAP is off. This port mirrored
/// the field and answered a non-nil value with `fatalError("clap_emb not
/// implemented")`, so the only thing the option could do was crash the host app.
/// No checkpoint published for this model sets it. It is not modelled here;
/// passing one is now a compile error rather than a runtime abort, and the
/// encoder still concatenates the zero-width CLAP slot so the sequence layout is
/// unchanged.
public struct T3Cond {
    public var speaker_emb: MLXArray
    public var cond_prompt_speech_tokens: MLXArray? = nil
    public var cond_prompt_speech_emb: MLXArray? = nil
    public var emotion_adv: MLXArray? = nil

    public init(
        speaker_emb: MLXArray,
        cond_prompt_speech_tokens: MLXArray? = nil,
        cond_prompt_speech_emb: MLXArray? = nil,
        emotion_adv: MLXArray? = nil
    ) {
        self.speaker_emb = speaker_emb
        self.cond_prompt_speech_tokens = cond_prompt_speech_tokens
        self.cond_prompt_speech_emb = cond_prompt_speech_emb
        self.emotion_adv = emotion_adv ?? MLXArray(0.5)
    }
}

public final class T3CondEnc: Module {
    public let hp: T3Config

    @ModuleInfo var spkr_enc: Linear
    @ModuleInfo var emotion_adv_fc: Linear?
    @ModuleInfo var perceiver: Perceiver?

    public init(_ hp: T3Config) {
        self.hp = hp

        if hp.encoder_type == "voice_encoder" {
            self._spkr_enc.wrappedValue = Linear(hp.speaker_embed_size, hp.n_channels)
        } else {
            fatalError("Unsupported encoder_type: \(hp.encoder_type)")
        }

        if hp.emotion_adv {
            self._emotion_adv_fc.wrappedValue = Linear(1, hp.n_channels, bias: false)
        } else {
            self._emotion_adv_fc.wrappedValue = nil
        }

        if hp.use_perceiver_resampler {
            self._perceiver.wrappedValue = Perceiver()
        } else {
            self._perceiver.wrappedValue = nil
        }

        super.init()
    }

    public func callAsFunction(_ cond: inout T3Cond) -> MLXArray {
        let B = cond.speaker_emb.dim(0)
        var cond_spkr = spkr_enc(cond.speaker_emb.reshaped([B, hp.speaker_embed_size]))
        cond_spkr = expandDims(cond_spkr, axis: 1)

        let empty = cond_spkr[0..., 0..<0, 0...]

        // Zero-width, always: see T3Cond on why CLAP conditioning is not modelled.
        // Kept in the concatenation because it is where upstream's layout puts it.
        let cond_clap = empty

        var cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb == nil {
            cond_prompt_speech_emb = empty
        } else if let perceiver = perceiver {
            cond_prompt_speech_emb = perceiver(cond_prompt_speech_emb!)
        }

        var cond_emotion_adv = empty
        if hp.emotion_adv {
            guard let emotion = cond.emotion_adv else {
                fatalError("emotion_adv must be provided when enabled")
            }
            var emotionVal = emotion
            if emotionVal.ndim == 0 {
                emotionVal = emotionVal.reshaped([1, 1, 1])
            } else if emotionVal.ndim == 1 {
                emotionVal = emotionVal.reshaped([-1, 1, 1])
            } else if emotionVal.ndim == 2 {
                emotionVal = expandDims(emotionVal, axis: -1)
            }
            cond_emotion_adv = emotion_adv_fc!(emotionVal)
        }

        let prompt = cond_prompt_speech_emb ?? empty
        return MLX.concatenated([cond_spkr, cond_clap, prompt, cond_emotion_adv], axis: 1)
    }

    func debugComponents(_ cond: inout T3Cond) -> [String: MLXArray] {
        let B = cond.speaker_emb.dim(0)
        var cond_spkr = spkr_enc(cond.speaker_emb.reshaped([B, hp.speaker_embed_size]))
        cond_spkr = expandDims(cond_spkr, axis: 1)

        let empty = cond_spkr[0..., 0..<0, 0...]
        let cond_clap = empty

        var cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        let cond_prompt_raw = cond_prompt_speech_emb ?? empty
        var perceiverDebug: [String: MLXArray] = [:]
        if cond_prompt_speech_emb == nil {
            cond_prompt_speech_emb = empty
        } else if let perceiver = perceiver {
            perceiverDebug = perceiver.debugForward(cond_prompt_speech_emb!)
            if let out = perceiverDebug["t3_perceiver_out"] {
                cond_prompt_speech_emb = out
            } else {
                cond_prompt_speech_emb = perceiver(cond_prompt_speech_emb!)
            }
        }

        var cond_emotion_adv = empty
        if hp.emotion_adv {
            guard let emotion = cond.emotion_adv else {
                fatalError("emotion_adv must be provided when enabled")
            }
            var emotionVal = emotion
            if emotionVal.ndim == 0 {
                emotionVal = emotionVal.reshaped([1, 1, 1])
            } else if emotionVal.ndim == 1 {
                emotionVal = emotionVal.reshaped([-1, 1, 1])
            } else if emotionVal.ndim == 2 {
                emotionVal = expandDims(emotionVal, axis: -1)
            }
            cond_emotion_adv = emotion_adv_fc!(emotionVal)
        }

        let prompt = cond_prompt_speech_emb ?? empty
        let cond_emb = MLX.concatenated([cond_spkr, cond_clap, prompt, cond_emotion_adv], axis: 1)
        var outputs: [String: MLXArray] = [
            "t3_cond_spkr": cond_spkr,
            "t3_cond_prompt_raw": cond_prompt_raw,
            "t3_cond_prompt": prompt,
            "t3_cond_emotion": cond_emotion_adv,
            "t3_cond_emb": cond_emb,
        ]
        if !perceiverDebug.isEmpty {
            outputs.merge(perceiverDebug, uniquingKeysWith: { current, _ in current })
        }
        if let perceiver = perceiver {
            outputs["t3_perceiver_query"] = perceiver.debugQuery()
        }
        return outputs
    }
}
