import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

public final class T3: Module {
    public let hp: T3Config
    public let cfg: LlamaConfig

    @ModuleInfo var tfmr: LlamaModel
    @ModuleInfo var cond_enc: T3CondEnc
    @ModuleInfo var text_emb: Embedding
    @ModuleInfo var speech_emb: Embedding
    @ModuleInfo var text_pos_emb: LearnedPositionEmbeddings?
    @ModuleInfo var speech_pos_emb: LearnedPositionEmbeddings?
    @ModuleInfo var text_head: Linear
    @ModuleInfo var speech_head: Linear

    public init(_ hp: T3Config? = nil) {
        let hp = hp ?? .english_only()
        self.hp = hp
        guard let cfg = LLAMA_CONFIGS[hp.llama_config_name] else {
            fatalError("Unknown Llama config: \(hp.llama_config_name)")
        }
        self.cfg = cfg

        self._tfmr.wrappedValue = LlamaModel(cfg)
        self._cond_enc.wrappedValue = T3CondEnc(hp)
        self._text_emb.wrappedValue = Embedding(
            embeddingCount: hp.text_tokens_dict_size,
            dimensions: cfg.hidden_size
        )
        self._speech_emb.wrappedValue = Embedding(
            embeddingCount: hp.speech_tokens_dict_size,
            dimensions: cfg.hidden_size
        )

        if hp.input_pos_emb == "learned" {
            let maxText = hp.max_text_tokens + 2
            let maxMel = hp.max_speech_tokens + 4
            self._text_pos_emb.wrappedValue = LearnedPositionEmbeddings(
                seqLen: maxText,
                modelDim: cfg.hidden_size
            )
            self._speech_pos_emb.wrappedValue = LearnedPositionEmbeddings(
                seqLen: maxMel,
                modelDim: cfg.hidden_size
            )
        } else {
            self._text_pos_emb.wrappedValue = nil
            self._speech_pos_emb.wrappedValue = nil
        }

        self._text_head.wrappedValue = Linear(cfg.hidden_size, hp.text_tokens_dict_size, bias: false)
        self._speech_head.wrappedValue = Linear(cfg.hidden_size, hp.speech_tokens_dict_size, bias: false)

        super.init()

        let rope = PyTorchCompatibleRoPE(
            dims: cfg.head_dim,
            base: cfg.rope_theta,
            maxSeqLen: cfg.max_position_embeddings,
            factor: 8.0,
            lowFreqFactor: 1.0,
            highFreqFactor: 4.0,
            oldContextLen: 8192
        )
        tfmr.replaceRoPE(rope)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("tfmr.") && !key.hasPrefix("tfmr.model.") {
                let newKey = key.replacingOccurrences(of: "tfmr.", with: "tfmr.model.")
                newWeights[newKey] = value
            } else {
                newWeights[key] = value
            }
        }
        return newWeights
    }

    public func prepare_conditioning(_ t3_cond: inout T3Cond) -> MLXArray {
        if t3_cond.cond_prompt_speech_tokens != nil && t3_cond.cond_prompt_speech_emb == nil {
            let tokens = t3_cond.cond_prompt_speech_tokens!
            let pos = speech_pos_emb!(tokens)
            t3_cond.cond_prompt_speech_emb = speech_emb(tokens) + pos
        }

        return cond_enc(&t3_cond)
    }

    public func prepare_input_embeds(
        t3_cond: inout T3Cond,
        text_tokens: MLXArray,
        speech_tokens: MLXArray,
        cfg_weight: Float = 0.0
    ) -> (MLXArray, Int) {
        let cond_emb = prepare_conditioning(&t3_cond)
        var textEmb = text_emb(text_tokens)

        if cfg_weight > 0.0 && textEmb.dim(0) > 1 {
            let zeroed = zerosLike(textEmb[1..<2, 0..., 0...])
            textEmb = MLX.concatenated([textEmb[0..<1, 0..., 0...], zeroed], axis: 0)
        }

        var speechEmb = speech_emb(speech_tokens)

        if hp.input_pos_emb == "learned" {
            if let textPos = text_pos_emb {
                textEmb = textEmb + textPos(text_tokens)
            }
            if let speechPos = speech_pos_emb {
                speechEmb = speechEmb + speechPos(speech_tokens)
            }
        }

        var cond = cond_emb
        if cond.dim(0) != textEmb.dim(0) {
            cond = broadcastTo(cond, [textEmb.dim(0), cond.dim(1), cond.dim(2)])
        }

        if speechEmb.dim(0) != textEmb.dim(0) {
            speechEmb = broadcastTo(speechEmb, [textEmb.dim(0), speechEmb.dim(1), speechEmb.dim(2)])
        }

        let embeds = MLX.concatenated([cond, textEmb, speechEmb], axis: 1)
        return (embeds, cond.dim(1))
    }

    public func debugInputEmbeddings(
        t3_cond: inout T3Cond,
        text_tokens: MLXArray,
        cfg_weight: Float = 0.0
    ) -> [String: MLXArray] {
        let bosToken = MLXArray([hp.start_speech_token], [1, 1])
        let (embeds, _) = prepare_input_embeds(
            t3_cond: &t3_cond,
            text_tokens: text_tokens,
            speech_tokens: bosToken,
            cfg_weight: cfg_weight
        )

        var bosEmbed = speech_emb(bosToken)
        if let pos = speech_pos_emb {
            bosEmbed = bosEmbed + pos.getFixedEmbedding(0)
        }
        if cfg_weight > 0.0 {
            bosEmbed = MLX.concatenated([bosEmbed, bosEmbed], axis: 0)
        }

        var condEmb = prepare_conditioning(&t3_cond)
        var textEmb = text_emb(text_tokens)
        if cfg_weight > 0.0 {
            let zeroed = zerosLike(textEmb[0..<1, 0..., 0...])
            textEmb = MLX.concatenated([textEmb[0..<1, 0..., 0...], zeroed], axis: 0)
        }
        if hp.input_pos_emb == "learned", let textPos = text_pos_emb {
            textEmb = textEmb + textPos(text_tokens)
        }
        if condEmb.dim(0) != textEmb.dim(0) {
            condEmb = broadcastTo(condEmb, [textEmb.dim(0), condEmb.dim(1), condEmb.dim(2)])
        }

        let inputEmbeddings = MLX.concatenated([embeds, bosEmbed], axis: 1)
        return [
            "t3_cond_emb": condEmb,
            "t3_text_emb": textEmb,
            "t3_embeds": embeds,
            "t3_bos_embed": bosEmbed,
            "t3_input_embeddings": inputEmbeddings,
        ]
    }

    public func debugPrefillHidden(
        t3_cond: inout T3Cond,
        text_tokens: MLXArray,
        cfg_weight: Float = 0.0
    ) -> [String: MLXArray] {
        var debug = debugInputEmbeddings(
            t3_cond: &t3_cond,
            text_tokens: text_tokens,
            cfg_weight: cfg_weight
        )
        guard let inputEmbeddings = debug["t3_input_embeddings"] else {
            return debug
        }

        let hidden = tfmr(
            inputs: nil,
            mask: .causal,
            cache: nil,
            input_embeddings: inputEmbeddings
        )
        debug["t3_prefill_hidden"] = hidden
        return debug
    }

    public func debugLayer0(
        input_embeddings: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal
    ) -> [String: MLXArray] {
        tfmr.debugLayer0(input_embeddings: input_embeddings, mask: mask)
    }

    public func debugConditioningComponents(t3_cond: inout T3Cond) -> [String: MLXArray] {
        if t3_cond.cond_prompt_speech_tokens != nil && t3_cond.cond_prompt_speech_emb == nil {
            let tokens = t3_cond.cond_prompt_speech_tokens!
            let pos = speech_pos_emb!(tokens)
            t3_cond.cond_prompt_speech_emb = speech_emb(tokens) + pos
        }
        return cond_enc.debugComponents(&t3_cond)
    }

    public func callAsFunction(
        t3_cond: inout T3Cond,
        text_tokens: MLXArray,
        text_token_lens: MLXArray,
        speech_tokens: MLXArray,
        speech_token_lens: MLXArray,
        cache: [KVCache]? = nil
    ) -> [String: MLXArray] {
        let (embeds, lenCond) = prepare_input_embeds(
            t3_cond: &t3_cond,
            text_tokens: text_tokens,
            speech_tokens: speech_tokens
        )

        let hidden = tfmr(
            inputs: nil,
            mask: .causal,
            cache: cache,
            input_embeddings: embeds
        )

        let B = text_tokens.dim(0)
        let lenText = text_tokens.dim(1)
        let lenSpeech = speech_tokens.dim(1)
        let dim = hidden.dim(2)

        var textLatents = MLXArray.zeros([B, lenText, dim])
        var speechLatents = MLXArray.zeros([B, lenSpeech, dim])

        for i in 0..<B {
            let ttl = text_token_lens[i].item(Int.self)
            let stl = speech_token_lens[i].item(Int.self)

            let textStart = lenCond
            let textEnd = lenCond + ttl

            let speechStart = lenCond + lenText
            let speechEnd = speechStart + stl

            if ttl > 0 {
                textLatents[i, 0..<ttl, 0...] = hidden[i, textStart..<textEnd, 0...]
            }
            if stl > 0 {
                speechLatents[i, 0..<stl, 0...] = hidden[i, speechStart..<speechEnd, 0...]
            }
        }

        let textLogits = text_head(textLatents)
        let speechLogits = speech_head(speechLatents)

        return [
            "text_logits": textLogits,
            "text_latents": textLatents,
            "speech_logits": speechLogits,
            "speech_latents": speechLatents,
            "hidden_states": hidden
        ]
    }

    public func inference(
        t3_cond: inout T3Cond,
        text_tokens: MLXArray,
        initial_speech_tokens: MLXArray? = nil,
        max_new_tokens: Int? = nil,
        temperature: Float = 0.8,
        top_p: Float = 1.0,
        min_p: Float = 0.05,
        repetition_penalty: Float = 1.2,
        cfg_weight: Float = 0.5,
        language_id: String? = nil
    ) throws -> MLXArray {
        try Task.checkCancellation()
        var textTokens = text_tokens
        if textTokens.ndim == 1 {
            textTokens = expandDims(textTokens, axis: 0)
        }

        let textLen = textTokens.dim(1)
        let maxNewTokens: Int = {
            if let max = max_new_tokens {
                return max
            }
            let estimate = textLen * 5
            return min(300, max(100, Int(Double(estimate) * 1.5)))
        }()
        let expectedTokens = textLen * 5

        let bosToken = MLXArray([hp.start_speech_token], [1, 1])

        do {
            let (embedsTmp, _) = prepare_input_embeds(
                t3_cond: &t3_cond,
                text_tokens: textTokens,
                speech_tokens: bosToken,
                cfg_weight: cfg_weight
            )
            var embedsLocal = embedsTmp
            var bosEmbed = speech_emb(bosToken)
            if let pos = speech_pos_emb {
                bosEmbed = bosEmbed + pos.getFixedEmbedding(0)
            }
            if cfg_weight > 0.0 {
                bosEmbed = MLX.concatenated([bosEmbed, bosEmbed], axis: 0)
            }

            var condEmb = prepare_conditioning(&t3_cond)
            var textEmb = text_emb(textTokens)

            if cfg_weight > 0.0 {
                let zeroed = zerosLike(textEmb[0..<1, 0..., 0...])
                textEmb = MLX.concatenated([textEmb[0..<1, 0..., 0...], zeroed], axis: 0)
            }

            if hp.input_pos_emb == "learned", let textPos = text_pos_emb {
                textEmb = textEmb + textPos(textTokens)
            }

            if condEmb.dim(0) != textEmb.dim(0) {
                condEmb = broadcastTo(condEmb, [textEmb.dim(0), condEmb.dim(1), condEmb.dim(2)])
            }

            let inputEmbeddings = MLX.concatenated([embedsLocal, bosEmbed], axis: 1)
            let prefixLen = inputEmbeddings.dim(1)

            // Pre-allocate KV cache to avoid dynamic growth during generation
            let maxCacheSize = prefixLen + maxNewTokens + 16  // buffer for safety
            var cache = (0..<cfg.num_hidden_layers).map { _ in KVCacheSimple(maxSize: maxCacheSize) as KVCache }

            var generatedIds: [Int] = [hp.start_speech_token]

            var hidden = tfmr(
                inputs: nil,
                mask: .causal,
                cache: cache,
                input_embeddings: inputEmbeddings
            )

            var currentLen = prefixLen

            for step in 0..<maxNewTokens {
                try Task.checkCancellation()
                let lastIndex = hidden.dim(1) - 1
                var logits = speech_head(hidden[0..., lastIndex..<lastIndex + 1, 0...]).squeezed(axis: 1)

                if cfg_weight > 0.0 && logits.dim(0) > 1 {
                    let condLogits = logits[0..<1, 0...]
                    let uncondLogits = logits[1..<2, 0...]
                    logits = condLogits + cfg_weight * (condLogits - uncondLogits)
                } else {
                    logits = logits[0..<1, 0...]
                }

                logits = applyRepetitionPenalty(logits: logits, tokens: generatedIds, penalty: repetition_penalty, contextSize: maxNewTokens)

                if step > expectedTokens {
                    let eosBoost = min(20.0, 1.0 + Float(step - expectedTokens) * 0.5)
                    let current = logits[0..., hp.stop_speech_token]
                    logits[0..., hp.stop_speech_token] = current + MLX.log(MLXArray(eosBoost))
                }

                let maxReasonableFrames: Int
                if language_id == "en" {
                    maxReasonableFrames = Int(Float(2.5) * Float(textTokens.dim(1)))
                } else {
                    maxReasonableFrames = 2 * textTokens.dim(1)
                }
                if step > maxReasonableFrames {
                    let vocab = logits.dim(-1)
                    var forced = full([1, vocab], values: Float(-32768.0))
                    forced[0, hp.stop_speech_token] = forced[0, hp.stop_speech_token] + MLXArray(65536.0)
                    logits = forced
                }

                if let last = generatedIds.last {
                    logits[0..., last] = logits[0..., last] + MLXArray(-5.0)
                }

                if temperature != 1.0 {
                    logits = logits / temperature
                }

                // Use softmax then log (MLX Swift doesn't have logsoftmax)
                // Note: log(softmax(x)) = x - logsumexp(x) for numerical stability
                // but we use the explicit form since MLX handles it efficiently
                var logprobs = MLX.log(softmax(logits, axis: -1))

                if min_p > 0.0 {
                    logprobs = applyMinP(logprobs: logprobs, minP: min_p, minTokensToKeep: 1)
                }

                if top_p > 0.0 && top_p < 1.0 {
                    logprobs = applyTopP(logprobs: logprobs, topP: top_p)
                }

                var probs = MLX.exp(logprobs)
                probs = probs / MLX.sum(probs, axis: -1, keepDims: true)

                let nextToken = categorical(MLX.log(probs))
                eval(nextToken)  // Synchronize for correct scalar extraction
                var nextTokenId = nextToken.item(Int.self)

                if let last = generatedIds.last, last == nextTokenId {
                    nextTokenId = hp.stop_speech_token
                }

                if nextTokenId == hp.stop_speech_token {
                    generatedIds.append(nextTokenId)
                    break
                }

                generatedIds.append(nextTokenId)

                var nextTokenArray = MLXArray([nextTokenId], [1, 1])
                var nextTokenEmbed = speech_emb(nextTokenArray)
                if let pos = speech_pos_emb {
                    nextTokenEmbed = nextTokenEmbed + pos.getFixedEmbedding(step + 1)
                }

                if cfg_weight > 0.0 {
                    nextTokenEmbed = MLX.concatenated([nextTokenEmbed, nextTokenEmbed], axis: 0)
                }

                currentLen += 1
                hidden = tfmr(
                    inputs: nil,
                    mask: .none,
                    cache: cache,
                    input_embeddings: nextTokenEmbed
                )
            }

            return MLXArray(generatedIds, [1, generatedIds.count])
        }
    }
}

private func applyRepetitionPenalty(
    logits: MLXArray,
    tokens: [Int],
    penalty: Float,
    contextSize: Int
) -> MLXArray {
    guard !tokens.isEmpty else { return logits }
    var logits = logits  // Explicit copy to avoid mutating input

    let recent = tokens.suffix(contextSize)
    let indices = MLXArray(recent.map { UInt32($0) })
    var selected = logits[0..., indices]
    selected = MLX.where(
        selected .< 0,
        selected * penalty,
        selected / penalty
    )
    logits[0..., indices] = selected
    return logits
}

private func applyMinP(logprobs: MLXArray, minP: Float, minTokensToKeep: Int) -> MLXArray {
    let sortedIndices = MLX.argSort(-logprobs, axis: -1)
    let sortedLogprobs = MLX.takeAlong(logprobs, sortedIndices, axis: -1)

    let topLogprobs = sortedLogprobs[0..., 0..<1]
    let scaledMinP = topLogprobs + MLXArray(Float(Foundation.log(Double(minP))))

    var tokensToRemove = sortedLogprobs .< scaledMinP
    tokensToRemove[0..., 0..<minTokensToKeep] = MLXArray(false)

    let selected = MLX.where(tokensToRemove, MLXArray(-Float.infinity), sortedLogprobs)

    let inverse = MLX.argSort(sortedIndices, axis: -1)
    return MLX.takeAlong(selected, inverse, axis: -1)
}

private func applyTopP(logprobs: MLXArray, topP: Float) -> MLXArray {
    let probs = MLX.exp(logprobs)
    // Sort descending (by negating) to get highest probability tokens first
    let sortedIndices = MLX.argSort(-logprobs, axis: -1)
    let sortedProbs = MLX.takeAlong(probs, sortedIndices, axis: -1)

    // Cumsum from highest to lowest probability
    let cumulative = MLX.cumsum(sortedProbs, axis: -1)
    
    // Create mask: keep tokens until cumulative reaches topP
    // Shift cumulative right by 1 to include the token that crosses threshold
    let cumulativeShifted = MLX.concatenated([
        MLXArray.zeros([cumulative.dim(0), 1]),
        cumulative[0..., 0..<(cumulative.dim(-1) - 1)]
    ], axis: -1)
    let sortedMask = cumulativeShifted .< MLXArray(topP)
    
    // Map mask back to original order
    let inverse = MLX.argSort(sortedIndices, axis: -1)
    let mask = MLX.takeAlong(sortedMask, inverse, axis: -1)

    return MLX.where(
        mask,
        logprobs,
        MLXArray(-Float.infinity)
    )
}
