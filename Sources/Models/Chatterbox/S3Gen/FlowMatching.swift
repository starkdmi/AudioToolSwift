// Copyright © 2025
// FlowMatching - Conditional Flow Matching for mel generation
// Pure MLX port of Python chatterbox/s3gen/flow_matching.py and matcha/flow_matching.py

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - CFM Parameters

/// Configuration for Conditional Flow Matching
public struct CFMParams {
    public var sigma_min: Float = 1e-6
    public var solver: String = "euler"
    public var t_scheduler: String = "cosine"
    public var training_cfg_rate: Float = 0.2
    public var inference_cfg_rate: Float = 0.7
    public var reg_loss_type: String = "l1"

    public init() {}
}

public let CFM_PARAMS = CFMParams()

// MARK: - Base CFM

/// Base Conditional Flow Matching module
public class BaseCFM: Module {
    public let n_feats: Int
    public let n_spks: Int
    public let spk_emb_dim: Int
    public let solver: String
    public let sigma_min: Float

    @ModuleInfo(key: "estimator") public var estimator: ConditionalDecoder

    public init(
        nFeats: Int,
        cfmParams: CFMParams,
        nSpks: Int = 1,
        spkEmbDim: Int = 128,
        estimator: ConditionalDecoder
    ) {
        self.n_feats = nFeats
        self.n_spks = nSpks
        self.spk_emb_dim = spkEmbDim
        self.solver = cfmParams.solver
        self.sigma_min = cfmParams.sigma_min
        self._estimator = ModuleInfo(wrappedValue: estimator)
        super.init()
    }

    /// Forward diffusion
    public func callAsFunction(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil
    ) -> MLXArray {
        let z = MLXRandom.normal(mu.shape) * temperature
        let tSpan = MLX.linspace(Float(0), Float(1), count: nTimesteps + 1)
        return solveEuler(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
    }

    /// Fixed Euler solver for ODEs
    public func solveEuler(
        x: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> MLXArray {
        var currentX = x
        var t = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        var sol: [MLXArray] = []
        for step in 1..<tSpan.shape[0] {
            let dphiDt = estimator(
                x: currentX,
                mask: mask,
                mu: mu,
                t: t,
                spks: spks,
                cond: cond,
                streaming: false
            )
            currentX = currentX + dt * dphiDt
            t = t + dt
            sol.append(currentX)

            if step < tSpan.shape[0] - 1 {
                dt = tSpan[step + 1] - t
            }
        }

        return sol.last ?? currentX
    }
}

// MARK: - Conditional CFM

/// Conditional Flow Matching with Classifier-Free Guidance
public class ConditionalCFM: BaseCFM {
    public let t_scheduler: String
    public let training_cfg_rate: Float
    public let inference_cfg_rate: Float

    public init(
        inChannels: Int,
        cfmParams: CFMParams,
        nSpks: Int = 1,
        spkEmbDim: Int = 64,
        estimator: ConditionalDecoder
    ) {
        self.t_scheduler = cfmParams.t_scheduler
        self.training_cfg_rate = cfmParams.training_cfg_rate
        self.inference_cfg_rate = cfmParams.inference_cfg_rate
        super.init(nFeats: inChannels, cfmParams: cfmParams, nSpks: nSpks, spkEmbDim: spkEmbDim, estimator: estimator)
    }

    /// Forward diffusion with optional caching
    public func callAsFunction(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        promptLen: Int = 0,
        flowCache: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var cache = flowCache ?? MLX.zeros([1, n_feats, 0, 2])
        var z = MLXRandom.normal(mu.shape) * temperature
        var muIn = mu
        let cacheSize = cache.shape[2]

        if cacheSize != 0 {
            let zCached = cache[0..., 0..., 0..., 0]
            let muCached = cache[0..., 0..., 0..., 1]
            z = concatenated([zCached, z[0..., 0..., cacheSize...]], axis: 2)
            muIn = concatenated([muCached, muIn[0..., 0..., cacheSize...]], axis: 2)
        }

        let tailStart = max(z.shape[2] - 34, 0)
        let zCache = concatenated([z[0..., 0..., 0..<promptLen], z[0..., 0..., tailStart...]], axis: 2)
        let muCache = concatenated([muIn[0..., 0..., 0..<promptLen], muIn[0..., 0..., tailStart...]], axis: 2)
        cache = stack([zCache, muCache], axis: -1)

        let result = solveEulerCFG(x: z, tScheduler: t_scheduler, nTimesteps: nTimesteps, mu: muIn, mask: mask, spks: spks, cond: cond)
        return (result, cache)
    }

    /// Euler solver with Classifier-Free Guidance
    public func solveEulerCFG(
        x: MLXArray,
        tScheduler: String,
        nTimesteps: Int,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> MLXArray {
        var tSpan = MLX.linspace(Float(0), Float(1), count: nTimesteps + 1)
        if tScheduler == "cosine" {
            tSpan = 1 - MLX.cos(tSpan * Float.pi * 0.5)
        }

        var currentX = x
        var t = tSpan[0].expandedDimensions(axis: 0)
        var dt = tSpan[1] - tSpan[0]

        let tLen = x.shape[2]
        var spksIn = MLX.zeros([2, spk_emb_dim])
        var condIn = MLX.zeros([2, n_feats, tLen])

        for step in 1..<tSpan.shape[0] {
            let xIn = concatenated([currentX, currentX], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)
            let muIn = concatenated([mu, MLX.zeros(like: mu)], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            if let spks = spks {
                spksIn = concatenated([spks, MLX.zeros(like: spks)], axis: 0)
            }
            if let cond = cond {
                condIn = concatenated([cond, MLX.zeros(like: cond)], axis: 0)
            }

            let dphiDt = estimator(
                x: xIn,
                mask: maskIn,
                mu: muIn,
                t: tIn,
                spks: spksIn,
                cond: condIn,
                streaming: false
            )

            let batchSize = currentX.shape[0]
            let dphiDtCond = dphiDt[0..<batchSize]
            let dphiDtUncond = dphiDt[batchSize...]
            let dphiDtCombined = (1.0 + inference_cfg_rate) * dphiDtCond - inference_cfg_rate * dphiDtUncond

            currentX = currentX + dt * dphiDtCombined
            
            // Use tSpan directly instead of accumulating to avoid precision issues
            if step < tSpan.shape[0] - 1 {
                t = tSpan[step].expandedDimensions(axis: 0)
                dt = tSpan[step + 1] - tSpan[step]
            }
        }

        return currentX
    }

    public func solveEulerCFGDebug(
        x: MLXArray,
        tScheduler: String,
        nTimesteps: Int,
        mu: MLXArray,
        mask: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> (MLXArray, [String: MLXArray]) {
        var tSpan = MLX.linspace(Float(0), Float(1), count: nTimesteps + 1)
        if tScheduler == "cosine" {
            tSpan = 1 - MLX.cos(tSpan * Float.pi * 0.5)
        }

        var currentX = x
        var t = tSpan[0].expandedDimensions(axis: 0)
        var dt = tSpan[1] - tSpan[0]

        let tLen = x.shape[2]
        var spksIn = MLX.zeros([2, spk_emb_dim])
        var condIn = MLX.zeros([2, n_feats, tLen])
        var debug: [String: MLXArray] = [
            "flow_t_span": tSpan
        ]

        for step in 1..<tSpan.shape[0] {
            let xIn = concatenated([currentX, currentX], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)
            let muIn = concatenated([mu, MLX.zeros(like: mu)], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            if let spks = spks {
                spksIn = concatenated([spks, MLX.zeros(like: spks)], axis: 0)
            }
            if let cond = cond {
                condIn = concatenated([cond, MLX.zeros(like: cond)], axis: 0)
            }

            let dphiDt = estimator(
                x: xIn,
                mask: maskIn,
                mu: muIn,
                t: tIn,
                spks: spksIn,
                cond: condIn,
                streaming: false
            )

            let batchSize = currentX.shape[0]
            let dphiDtCond = dphiDt[0..<batchSize]
            let dphiDtUncond = dphiDt[batchSize...]
            let dphiDtCombined = (1.0 + inference_cfg_rate) * dphiDtCond - inference_cfg_rate * dphiDtUncond

            let nextX = currentX + dt * dphiDtCombined
            if step == 1 {
                debug["flow_dphi_dt_step0"] = dphiDtCombined
                debug["flow_x_step0"] = nextX
            }

            currentX = nextX
            
            // Use tSpan directly instead of accumulating to avoid precision issues
            if step < tSpan.shape[0] - 1 {
                t = tSpan[step].expandedDimensions(axis: 0)
                dt = tSpan[step + 1] - tSpan[step]
            }
        }

        return (currentX, debug)
    }
}

// MARK: - Causal Conditional CFM

/// Causal Conditional Flow Matching with fixed noise
public class CausalConditionalCFM: ConditionalCFM {
    public static let MEL_CHANNELS = 80

    public override init(
        inChannels: Int = 240,
        cfmParams: CFMParams = CFM_PARAMS,
        nSpks: Int = 1,
        spkEmbDim: Int = 80,
        estimator: ConditionalDecoder
    ) {
        super.init(inChannels: inChannels, cfmParams: cfmParams, nSpks: nSpks, spkEmbDim: spkEmbDim, estimator: estimator)
    }

    /// Forward diffusion with noise for causal generation
    public func callAsFunction(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        streaming: Bool = false,
        noisedMels: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        var z = MLXRandom.normal(mu.shape) * temperature

        if let noise = noisedMels {
            let promptLen = mu.shape[2] - noise.shape[2]
            z = concatenated([z[0..., 0..., 0..<promptLen], noise], axis: 2)
        }

        var tSpan = MLX.linspace(Float(0), Float(1), count: nTimesteps + 1)
        if t_scheduler == "cosine" {
            tSpan = 1 - MLX.cos(tSpan * Float.pi * 0.5)
        }

        let result = solveEulerCFG(x: z, tScheduler: t_scheduler, nTimesteps: nTimesteps, mu: mu, mask: mask, spks: spks, cond: cond)
        return (result, nil)
    }

    public func callAsFunctionDebug(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        streaming: Bool = false,
        noisedMels: MLXArray? = nil
    ) -> (MLXArray, [String: MLXArray]) {
        var z = MLXRandom.normal(mu.shape) * temperature

        if let noise = noisedMels {
            let promptLen = mu.shape[2] - noise.shape[2]
            z = concatenated([z[0..., 0..., 0..<promptLen], noise], axis: 2)
        }

        let (result, debug) = solveEulerCFGDebug(
            x: z,
            tScheduler: t_scheduler,
            nTimesteps: nTimesteps,
            mu: mu,
            mask: mask,
            spks: spks,
            cond: cond
        )
        var merged = debug
        merged["flow_z"] = z
        return (result, merged)
    }
}
