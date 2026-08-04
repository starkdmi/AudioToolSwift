import MLX
import MLXNN
import Foundation

// MARK: - HTDemucs Configuration

/// Configuration for HTDemucs model
public struct HTDemucsConfig {
    public var sources: [String] = ["drums", "bass", "other", "vocals"]
    public var audioChannels: Int = 2
    public var channels: Int = 48
    public var channelsTime: Int? = nil
    public var growth: Int = 2
    public var nfft: Int = 4096
    public var wienerIters: Int = 0
    public var endIters: Int = 0
    public var wienerResidual: Bool = false
    public var cac: Bool = true
    public var depth: Int = 4
    public var rewrite: Bool = true
    public var multiFreqs: [Int]? = nil
    public var multiFreqsDepth: Int = 3
    public var freqEmb: Float = 0.2
    public var embScale: Float = 10
    public var embSmooth: Bool = true
    public var kernelSize: Int = 8
    public var timeStride: Int = 2
    public var stride: Int = 4
    public var context: Int = 1
    public var contextEnc: Int = 0
    public var normStarts: Int = 4
    public var normGroups: Int = 4
    public var dconvMode: Int = 3
    public var dconvDepth: Int = 2
    public var dconvComp: Float = 8
    public var dconvInit: Float = 1e-3
    public var bottomChannels: Int = 512
    public var tLayers: Int = 5
    public var tEmb: String = "sin"
    public var tHiddenScale: Float = 4.0
    public var tHeads: Int = 8
    public var tDropout: Float = 0.02
    public var tMaxPositions: Int = 10000
    public var tNormIn: Bool = true
    public var tNormInGroup: Bool = false
    public var tGroupNorm: Bool = false
    public var tNormFirst: Bool = true
    public var tNormOut: Bool = true
    public var tMaxPeriod: Float = 10000.0
    public var tWeightDecay: Float = 0.05
    public var tLr: Float? = nil
    public var tLayerScale: Bool = true
    public var tGelu: Bool = true
    public var tWeightPosEmbed: Float = 1.0
    public var tCrossFirst: Bool = false
    public var rescale: Float = 0.1
    public var samplerate: Int = 44100
    public var segment: Float = 7.8
    public var useTrainSegment: Bool = true
    
    public init() {}
}

// MARK: - HTDemucs Model

/// Hybrid Transformer Demucs model for audio source separation
public class HTDemucs: Module {
    let config: HTDemucsConfig
    let hopLength: Int
    
    // Encoder layers (up to 4 depth)
    @ModuleInfo var encoder_0: HEncLayer?
    @ModuleInfo var encoder_1: HEncLayer?
    @ModuleInfo var encoder_2: HEncLayer?
    @ModuleInfo var encoder_3: HEncLayer?
    
    // Time encoder layers
    @ModuleInfo var tencoder_0: HEncLayer?
    @ModuleInfo var tencoder_1: HEncLayer?
    @ModuleInfo var tencoder_2: HEncLayer?
    @ModuleInfo var tencoder_3: HEncLayer?
    
    // Decoder layers
    @ModuleInfo var decoder_0: HDecLayer?
    @ModuleInfo var decoder_1: HDecLayer?
    @ModuleInfo var decoder_2: HDecLayer?
    @ModuleInfo var decoder_3: HDecLayer?
    
    // Time decoder layers
    @ModuleInfo var tdecoder_0: HDecLayer?
    @ModuleInfo var tdecoder_1: HDecLayer?
    @ModuleInfo var tdecoder_2: HDecLayer?
    @ModuleInfo var tdecoder_3: HDecLayer?
    
    // Frequency embedding
    @ModuleInfo var freq_emb: ScaledEmbedding?
    let freq_emb_scale: Float
    
    // Channel up/downsampling
    @ModuleInfo var channel_upsampler: Conv1d?
    @ModuleInfo var channel_downsampler: Conv1d?
    @ModuleInfo var channel_upsampler_t: Conv1d?
    @ModuleInfo var channel_downsampler_t: Conv1d?
    
    // Cross-transformer
    @ModuleInfo var crosstransformer: CrossTransformerEncoder?
    
    // Track which encoder layers are "empty" (for injection)
    var tencoderEmpty: [Bool] = []
    var tdecoderEmpty: [Bool] = []
    
    public init(config: HTDemucsConfig = HTDemucsConfig()) {
        self.config = config
        self.hopLength = config.nfft / 4
        self.freq_emb_scale = config.freqEmb
        
        var chin = config.audioChannels
        var chinZ = chin
        if config.cac {
            chinZ *= 2
        }
        
        var chout = config.channelsTime ?? config.channels
        var choutZ = config.channels
        var freqs = config.nfft / 2
        
        // Track encoder info for building decoders
        var encoderChins: [Int] = []
        var encoderChouts: [Int] = []
        var tencoderChins: [Int] = []
        var tencoderChouts: [Int] = []
        var freqsAtIdx: [Int] = []
        
        var freqEmbFreqs: Int? = nil
        var freqEmbChin: Int? = nil
        
        // Track decoder output channels - these are what each decoder outputs
        var decoderChouts: [Int] = []
        var tdecoderChouts: [Int] = []
        
        for index in 0..<config.depth {
            let norm = index >= config.normStarts
            let freq = freqs > 1
            var stri = config.stride
            var ker = config.kernelSize
            
            if !freq {
                ker = config.timeStride * 2
                stri = config.timeStride
            }
            
            var padKw = true
            var lastFreq = false
            if freq && freqs <= config.kernelSize {
                ker = freqs
                padKw = false
                lastFreq = true
            }
            
            let dconvKw = (depth: config.dconvDepth, compress: config.dconvComp, init_val: config.dconvInit, gelu: true)
            
            if lastFreq {
                choutZ = max(chout, choutZ)
                chout = choutZ
            }
            
            // Save for decoder building
            encoderChins.append(chinZ)
            encoderChouts.append(choutZ)
            tencoderChins.append(chin)
            tencoderChouts.append(chout)
            freqsAtIdx.append(freqs)
            
            // Calculate decoder output channels
            // For index 0, the decoder outputs to sources, so chinZ is updated
            if index == 0 {
                let decoderOutChin = config.audioChannels * config.sources.count * (config.cac ? 2 : 1)
                decoderChouts.append(decoderOutChin)
            } else {
                decoderChouts.append(chinZ)
            }
            
            // Create encoder
            let enc = HEncLayer(
                chin: chinZ,
                chout: choutZ,
                kernel_size: ker,
                stride: stri,
                norm_groups: config.normGroups,
                empty: false,
                freq: freq,
                dconv: (config.dconvMode & 1) != 0,
                norm: norm,
                context: config.contextEnc,
                dconv_depth: dconvKw.depth,
                dconv_compress: dconvKw.compress,
                dconv_init: dconvKw.init_val,
                dconv_gelu: dconvKw.gelu,
                pad: padKw,
                rewrite: config.rewrite
            )
            
            switch index {
            case 0: self.encoder_0 = enc
            case 1: self.encoder_1 = enc
            case 2: self.encoder_2 = enc
            case 3: self.encoder_3 = enc
            default: break
            }
            
            // Create time encoder
            if freq {
                let tenc = HEncLayer(
                    chin: chin,
                    chout: chout,
                    kernel_size: config.kernelSize,
                    stride: config.stride,
                    norm_groups: config.normGroups,
                    empty: lastFreq,
                    freq: false,
                    dconv: (config.dconvMode & 1) != 0,
                    norm: norm,
                    context: config.contextEnc,
                    dconv_depth: dconvKw.depth,
                    dconv_compress: dconvKw.compress,
                    dconv_init: dconvKw.init_val,
                    dconv_gelu: dconvKw.gelu,
                    pad: true,
                    rewrite: config.rewrite
                )
                tencoderEmpty.append(lastFreq)
                
                // Calculate time decoder output channels
                if index == 0 {
                    let tdecoderOutChin = config.audioChannels * config.sources.count
                    tdecoderChouts.append(tdecoderOutChin)
                } else {
                    tdecoderChouts.append(chin)
                }
                
                switch index {
                case 0: self.tencoder_0 = tenc
                case 1: self.tencoder_1 = tenc
                case 2: self.tencoder_2 = tenc
                case 3: self.tencoder_3 = tenc
                default: break
                }
            }
            
            if index == 0 {
                chin = config.audioChannels * config.sources.count
                chinZ = chin
                if config.cac {
                    chinZ *= 2
                }
            }
            
            chin = chout
            chinZ = choutZ
            chout = chout * config.growth
            choutZ = choutZ * config.growth
            
            if freq {
                if freqs <= config.kernelSize {
                    freqs = 1
                } else {
                    freqs /= config.stride
                }
            }
            
            if index == 0 && config.freqEmb > 0 {
                freqEmbFreqs = freqs
                freqEmbChin = chinZ
            }
        }
        
        // Frequency embedding
        if let freqs = freqEmbFreqs, let chinEmb = freqEmbChin {
            self.freq_emb = ScaledEmbedding(numEmbeddings: freqs, embeddingDim: chinEmb, scale: config.embScale, smooth: config.embSmooth)
        }
        
        // Build decoders in reverse
        let numEncoders = config.depth
        let numTEncoders = tencoderEmpty.count
        
        for index in 0..<numEncoders {
            let revIdx = numEncoders - 1 - index
            let decoderChin = encoderChouts[revIdx]
            let decoderChout = decoderChouts[revIdx]  // Use pre-calculated decoder output channels
            
            let freqAtIdx = freqsAtIdx[revIdx]
            let freq = freqAtIdx > 1
            var ker = config.kernelSize
            var stri = config.stride
            var padKw = true
            
            if !freq {
                ker = config.timeStride * 2
                stri = config.timeStride
            }
            
            if freq && freqAtIdx <= config.kernelSize {
                ker = freqAtIdx
                padKw = false
            }
            
            let norm = revIdx >= config.normStarts
            let dconvKw = (depth: config.dconvDepth, compress: config.dconvComp, init_val: config.dconvInit, gelu: true)
            
            let dec = HDecLayer(
                chin: decoderChin,
                chout: decoderChout,
                last: revIdx == 0,
                kernel_size: ker,
                stride: stri,
                norm_groups: config.normGroups,
                empty: false,
                freq: freq,
                dconv: (config.dconvMode & 2) != 0,
                norm: norm,
                context: config.context,
                dconv_depth: dconvKw.depth,
                dconv_compress: dconvKw.compress,
                dconv_init: dconvKw.init_val,
                dconv_gelu: dconvKw.gelu,
                pad: padKw,
                context_freq: true,
                rewrite: config.rewrite
            )
            
            switch index {
            case 0: self.decoder_0 = dec
            case 1: self.decoder_1 = dec
            case 2: self.decoder_2 = dec
            case 3: self.decoder_3 = dec
            default: break
            }
        }
        
        // Build time decoders in reverse
        for index in 0..<numTEncoders {
            let revIdx = numTEncoders - 1 - index
            let decoderChin = tencoderChouts[revIdx]
            let decoderChout = tdecoderChouts[revIdx]  // Use pre-calculated time decoder output channels
            let isEmpty = tencoderEmpty[revIdx]
            
            let tdec = HDecLayer(
                chin: decoderChin,
                chout: decoderChout,
                last: revIdx == 0,
                kernel_size: config.kernelSize,
                stride: config.stride,
                norm_groups: config.normGroups,
                empty: isEmpty,
                freq: false,
                dconv: (config.dconvMode & 2) != 0,
                norm: revIdx >= config.normStarts,
                context: config.context,
                dconv_depth: config.dconvDepth,
                dconv_compress: config.dconvComp,
                dconv_init: config.dconvInit,
                dconv_gelu: true,
                pad: true,
                context_freq: true,
                rewrite: config.rewrite
            )
            tdecoderEmpty.append(isEmpty)
            
            switch index {
            case 0: self.tdecoder_0 = tdec
            case 1: self.tdecoder_1 = tdec
            case 2: self.tdecoder_2 = tdec
            case 3: self.tdecoder_3 = tdec
            default: break
            }
        }
        
        // Transformer
        let transformerChannels: Int
        var finalTransformerChannels = config.channels
        for _ in 0..<(config.depth - 1) {
            finalTransformerChannels *= config.growth
        }
        
        if config.bottomChannels > 0 {
            self.channel_upsampler = Conv1d(inputChannels: finalTransformerChannels, outputChannels: config.bottomChannels, kernelSize: 1)
            self.channel_downsampler = Conv1d(inputChannels: config.bottomChannels, outputChannels: finalTransformerChannels, kernelSize: 1)
            self.channel_upsampler_t = Conv1d(inputChannels: finalTransformerChannels, outputChannels: config.bottomChannels, kernelSize: 1)
            self.channel_downsampler_t = Conv1d(inputChannels: config.bottomChannels, outputChannels: finalTransformerChannels, kernelSize: 1)
            transformerChannels = config.bottomChannels
        } else {
            transformerChannels = finalTransformerChannels
        }
        
        if config.tLayers > 0 {
            self.crosstransformer = CrossTransformerEncoder(
                dim: transformerChannels,
                emb: config.tEmb,
                hiddenScale: config.tHiddenScale,
                numHeads: config.tHeads,
                numLayers: config.tLayers,
                crossFirst: config.tCrossFirst,
                dropout: config.tDropout,
                maxPositions: config.tMaxPositions,
                normIn: config.tNormIn,
                normInGroup: config.tNormInGroup,
                groupNorm: config.tGroupNorm,
                normFirst: config.tNormFirst,
                normOut: config.tNormOut,
                maxPeriod: config.tMaxPeriod,
                weightDecay: config.tWeightDecay,
                lr: config.tLr,
                layerScale: config.tLayerScale,
                gelu: config.tGelu,
                weightPosEmbed: config.tWeightPosEmbed,
                rescale: config.rescale
            )
        }
        
        super.init()
    }
    
    // Helper to get encoder by index
    func getEncoder(_ idx: Int) -> HEncLayer? {
        switch idx {
        case 0: return encoder_0
        case 1: return encoder_1
        case 2: return encoder_2
        case 3: return encoder_3
        default: return nil
        }
    }
    
    // Helper to get time encoder by index
    func getTEncoder(_ idx: Int) -> HEncLayer? {
        switch idx {
        case 0: return tencoder_0
        case 1: return tencoder_1
        case 2: return tencoder_2
        case 3: return tencoder_3
        default: return nil
        }
    }
    
    // Helper to get decoder by index
    func getDecoder(_ idx: Int) -> HDecLayer? {
        switch idx {
        case 0: return decoder_0
        case 1: return decoder_1
        case 2: return decoder_2
        case 3: return decoder_3
        default: return nil
        }
    }
    
    // Helper to get time decoder by index
    func getTDecoder(_ idx: Int) -> HDecLayer? {
        switch idx {
        case 0: return tdecoder_0
        case 1: return tdecoder_1
        case 2: return tdecoder_2
        case 3: return tdecoder_3
        default: return nil
        }
    }
    
    func specTransform(_ x: MLXArray) -> MLXArray {
        // x is (B, C, T)
        let hl = hopLength
        let nfft = config.nfft
        
        // Pad for alignment
        let le = Int(ceil(Double(x.shape[2]) / Double(hl)))
        let pad = hl / 2 * 3
        let padRight = pad + le * hl - x.shape[2]
        
        // Reflect padding
        let xPadded = pad1d(x, paddings: (pad, padRight), mode: "reflect", axis: 2)
        
        var z = spectro(xPadded, nFFT: nfft, hopLength: hl, pad: 0)  // (B, C, F, T) complex
        z = z[0..., 0..., ..<(-1), 0...]  // Remove last freq bin
        
        // Slice to remove padding effects
        z = z[0..., 0..., 0..., 2..<(2 + le)]
        
        return z
    }
    
    func ispecTransform(_ z: MLXArray, length: Int) -> MLXArray {
        // z is (B, C, F, T) complex
        let hl = hopLength
        
        // Pad freq dim by 1 at end
        var zFreqPadded = MLX.padded(z, widths: [IntOrPair(0), IntOrPair(0), IntOrPair([0, 1]), IntOrPair(0)])
        
        // Pad time dim by 2 on each side
        let zTimePadded = MLX.padded(zFreqPadded, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair([2, 2])])
        
        // Compute expected output length
        let pad = hl / 2 * 3
        let le = hl * Int(ceil(Double(length) / Double(hl))) + 2 * pad
        
        // ISTFT
        var x = ispectro(zTimePadded, hopLength: hl, length: le, pad: 0)
        
        // Crop
        x = x[0..., 0..., pad..<(pad + length)]
        
        return x
    }
    
    public func callAsFunction(_ mix: MLXArray) -> MLXArray {
        // mix: (Batch, Channels, Time)
        var inputMix = mix
        let length = inputMix.shape[2]
        var lengthPrePad: Int? = nil
        
        // Handle segment padding
        if config.useTrainSegment {
            let trainingLength = Int(config.segment * Float(config.samplerate))
            if inputMix.shape[2] < trainingLength {
                lengthPrePad = inputMix.shape[2]
                let padAmount = trainingLength - lengthPrePad!
                inputMix = MLX.padded(inputMix, widths: [IntOrPair(0), IntOrPair(0), IntOrPair([0, padAmount])])
            }
        }
        
        // 1. STFT on RAW mix
        let z = specTransform(inputMix)  // (B, C, F, T) complex
        let B = z.shape[0]
        let C = z.shape[1]
        let Freq = z.shape[2]
        let T = z.shape[3]

        
        // 2. Get magnitude for freq branch input
        var x: MLXArray
        if config.cac {
            // Stack real/imag as channels
            let zStacked = MLX.stacked([z.realPart(), z.imaginaryPart()], axis: 2)
            x = zStacked.reshaped([B, C * 2, Freq, T])

        } else {
            x = MLX.abs(z)

        }
        
        // 3. Normalize spectrogram
        let mean = x.mean(axes: [1, 2, 3], keepDims: true)
        let std = MLX.sqrt(x.variance(axes: [1, 2, 3], keepDims: true))
        x = (x - mean) / (MLXArray(Float(1e-5)) + std)

        
        // 4. Normalize waveform for time branch
        let xtRaw = inputMix
        let meant = xtRaw.mean(axes: [1, 2], keepDims: true)
        let stdt = MLX.sqrt(xtRaw.variance(axes: [1, 2], keepDims: true))
        let xtNorm = (xtRaw - meant) / (MLXArray(Float(1e-5)) + stdt)
        
        // Prepare time branch: (B, C, T) -> (B, T, C)
        var xt = xtNorm.transposed(0, 2, 1)
        
        // Prepare freq branch: (B, C, F, T) -> (B, T, F, C)
        x = x.transposed(0, 3, 2, 1)

        
        var saved: [MLXArray] = []
        var savedT: [MLXArray] = []
        var lengthsT: [Int] = []
        
        let numEncoders = config.depth
        let numTEncoders = tencoderEmpty.count
        
        // Encoder loop
        for idx in 0..<numEncoders {
            var inject: MLXArray? = nil
            
            if idx < numTEncoders {
                if let tenc = getTEncoder(idx) {
                    lengthsT.append(xt.shape[1])
                    xt = tenc(xt)
                    if !tencoderEmpty[idx] {
                        savedT.append(xt)
                    } else {
                        // inject xt into freq branch
                        inject = xt.expandedDimensions(axis: 2)
                    }
                }
            }
            
            if let enc = getEncoder(idx) {
                x = enc(x, inject: inject)
            }
            
            // Freq embedding
            if idx == 0, let freqEmb = freq_emb {
                let frs = MLXArray(0..<x.shape[2])
                let emb = freqEmb(frs)  // (F, C)
                x = x + freq_emb_scale * emb.expandedDimensions(axes: [0, 1])
            }
            
            saved.append(x)
        }
        
        // Transformer
        if let transformer = crosstransformer {
            // x is (B, T, F, C)
            var xTrans = x
            var xtTrans = xt
            
            if let upsampler = channel_upsampler, let upsamplerT = channel_upsampler_t {
                let Bx = xTrans.shape[0]
                let Tx = xTrans.shape[1]
                let Fx = xTrans.shape[2]
                let Cx = xTrans.shape[3]
                
                // Flatten to (B, F*T, C)
                xTrans = xTrans.transposed(0, 2, 1, 3).reshaped([Bx, Fx * Tx, Cx])
                xTrans = upsampler(xTrans)
                xTrans = xTrans.reshaped([Bx, Fx, Tx, -1]).transposed(0, 2, 1, 3)
                
                xtTrans = upsamplerT(xtTrans)
            }
            
            // Reshape x to (B, Fr, T, C) for transformer
            xTrans = xTrans.transposed(0, 2, 1, 3)
            
            (xTrans, xtTrans) = transformer(xTrans, xtTrans)
            
            // Reshape back to (B, T, F, C)
            xTrans = xTrans.transposed(0, 2, 1, 3)
            
            if let downsampler = channel_downsampler, let downsamplerT = channel_downsampler_t {
                let Bx = xTrans.shape[0]
                let Tx = xTrans.shape[1]
                let Fx = xTrans.shape[2]
                let Cx = xTrans.shape[3]
                
                xTrans = xTrans.transposed(0, 2, 1, 3).reshaped([Bx, Fx * Tx, Cx])
                xTrans = downsampler(xTrans)
                xTrans = xTrans.reshaped([Bx, Fx, Tx, -1]).transposed(0, 2, 1, 3)
                
                xtTrans = downsamplerT(xtTrans)
            }
            
            x = xTrans
            xt = xtTrans
        }
        

        
        // Decoder loop
        let offset = config.depth - tdecoderEmpty.count
        
        for idx in 0..<numEncoders {
            if let dec = getDecoder(idx) {
                var skip = saved.removeLast()
                
                // Trim x/skip to match in Freq dim
                let minFreq = min(x.shape[2], skip.shape[2])
                if x.shape[2] > minFreq {
                    x = centerTrim(x, length: minFreq, axis: 2)
                }
                if skip.shape[2] > minFreq {
                    skip = centerTrim(skip, length: minFreq, axis: 2)
                }
                
                let (xOut, pre) = dec(x, skip: skip)
                x = xOut
                
                // Handle time branch
                if idx >= offset {
                    let tdecIdx = idx - offset
                    if let tdec = getTDecoder(tdecIdx) {
                        let lengthT = lengthsT.removeLast()
                        
                        if tdecoderEmpty[tdecIdx] {
                            // Join freq/time
                            var preSq = pre ?? x
                            if preSq.shape[2] == 1 {
                                preSq = preSq.squeezed(axis: 2)
                            }
                            let (xtOut, _) = tdec(preSq, skip: nil, length: lengthT)
                            xt = xtOut
                        } else {
                            var skipT = savedT.removeLast()
                            
                            let minLen = min(xt.shape[1], skipT.shape[1])
                            if xt.shape[1] > minLen {
                                xt = centerTrim(xt, length: minLen, axis: 1)
                            }
                            if skipT.shape[1] > minLen {
                                skipT = centerTrim(skipT, length: minLen, axis: 1)
                            }
                            
                            let (xtOut, _) = tdec(xt, skip: skipT, length: lengthT)
                            xt = xtOut
                        }
                    }
                }
            }
        }
        

        
        // Output masking
        let S = config.sources.count
        let targetLength = length
        
        // Trim freq dim
        if x.shape[2] > Freq {
            x = centerTrim(x, length: Freq, axis: 2)
        }
        
        // Permute back to (B, C, F, T)
        x = x.transposed(0, 3, 2, 1)

        
        // x: (B, S*C_audio*2, F, T)
        x = x.reshaped([B, S, -1, Freq, T])
        
        // Denormalize freq branch
        x = x * std.expandedDimensions(axis: 1) + mean.expandedDimensions(axis: 1)
        
        // Convert to complex spectrogram (CAC mode)
        var xReshaped = x.reshaped([B, S, -1, 2, Freq, T])
        let xPermuted = xReshaped.transposed(0, 1, 2, 4, 5, 3)  // (B, S, C, Fr, T, 2)
        let realPart = xPermuted[0..., 0..., 0..., 0..., 0..., 0]
        let imagPart = xPermuted[0..., 0..., 0..., 0..., 0..., 1]
        let outSpecComplex = realPart + MLXArray(real: 0, imaginary: 1) * imagPart
        
        let outSpecGrouped = outSpecComplex.reshaped([B * S, -1, Freq, T])
        
        // ISTFT
        var wav = ispecTransform(outSpecGrouped, length: targetLength)
        wav = wav.reshaped([B, S, -1, wav.shape[2]])
        
        // Time branch output
        // xt: (B, T, S*C)
        let xtShape = xt.shape
        var xtReshaped = xt.reshaped([xtShape[0], xtShape[1], S, -1])  // (B, T, S, C)
        xtReshaped = xtReshaped.transposed(0, 2, 3, 1)  // (B, S, C, T)
        
        // Trim xt to match target length
        if xtReshaped.shape[3] > targetLength {
            xtReshaped = xtReshaped[0..., 0..., 0..., ..<targetLength]
        }
        
        xtReshaped = xtReshaped * stdt.expandedDimensions(axis: 1) + meant.expandedDimensions(axis: 1)
        
        var out = wav + xtReshaped
        
        if let prepadLen = lengthPrePad {
            out = out[0..., 0..., 0..., ..<prepadLen]
        }
        
        return out
    }
    

}


// MARK: - Weight Loading

extension HTDemucs {
    /// Convert Python-style key (encoder.0.conv.weight) to Swift-style (encoder_0.conv.weight)
    private static func convertPythonKeyToSwift(_ key: String) -> String {
        var result = key
        
        // Convert array indexing patterns like "encoder.0" to "encoder_0"
        // Handle: encoder, decoder, tencoder, tdecoder
        let patterns = ["encoder", "decoder", "tencoder", "tdecoder"]
        for pattern in patterns {
            let regex = try! NSRegularExpression(pattern: "\(pattern)\\.(\\d+)")
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "\(pattern)_$1")
        }
        
        // Convert crosstransformer.layers.X and crosstransformer.layers_t.X
        // Note: layers_t should be converted first to avoid double conversion
        let layersTRegex = try! NSRegularExpression(pattern: "crosstransformer\\.layers_t\\.(\\d+)")
        let layersTRange = NSRange(result.startIndex..., in: result)
        result = layersTRegex.stringByReplacingMatches(in: result, range: layersTRange, withTemplate: "crosstransformer.layers_t_$1")
        
        let layersRegex = try! NSRegularExpression(pattern: "crosstransformer\\.layers\\.(\\d+)")
        let layersRange = NSRange(result.startIndex..., in: result)
        result = layersRegex.stringByReplacingMatches(in: result, range: layersRange, withTemplate: "crosstransformer.layers_$1")
        
        // Convert DConv nested layers structure
        // Python: dconv.layers.D.layers.L -> Swift: dconv.{type}D_{idx}
        // Mapping for each depth D:
        //   layers.D.layers.0 -> conv{D}_0 (first Conv1d)
        //   layers.D.layers.1 -> norm{D}_0 (first GroupNorm)
        //   layers.D.layers.3 -> conv{D}_1 (second Conv1d)
        //   layers.D.layers.4 -> norm{D}_1 (second GroupNorm)
        //   layers.D.layers.6 -> scale{D} (LayerScale)
        
        // Handle dconv.layers.D.layers.L patterns
        let dconvMappings: [(String, String)] = [
            ("layers\\.(\\d+)\\.layers\\.0\\.", "conv$1_0."),  // first conv
            ("layers\\.(\\d+)\\.layers\\.1\\.", "norm$1_0."),  // first norm
            ("layers\\.(\\d+)\\.layers\\.3\\.", "conv$1_1."),  // second conv
            ("layers\\.(\\d+)\\.layers\\.4\\.", "norm$1_1."),  // second norm
            ("layers\\.(\\d+)\\.layers\\.6\\.", "scale$1."),   // scale
        ]
        
        for (pattern, replacement) in dconvMappings {
            let regex = try! NSRegularExpression(pattern: pattern)
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: replacement)
        }
        
        // Convert conv/conv_tr/rewrite based on freq vs time branch
        // encoder/decoder (freq branch) use 2d variants
        // tencoder/tdecoder (time branch) use 1d variants
        
        // For freq branches (encoder_X, decoder_X): conv -> conv2d, conv_tr -> conv_tr2d, rewrite -> rewrite2d
        if result.contains("encoder_") || result.contains("decoder_") {
            // Only convert if NOT tencoder/tdecoder
            if !result.hasPrefix("t") {
                result = result.replacingOccurrences(of: ".conv.", with: ".conv2d.")
                result = result.replacingOccurrences(of: ".conv_tr.", with: ".conv_tr2d.")
                result = result.replacingOccurrences(of: ".rewrite.", with: ".rewrite2d.")
            }
        }
        
        // For time branches (tencoder_X, tdecoder_X): conv -> conv1d, conv_tr -> conv_tr1d, rewrite -> rewrite1d
        if result.hasPrefix("tencoder_") || result.hasPrefix("tdecoder_") {
            result = result.replacingOccurrences(of: ".conv.", with: ".conv1d.")
            result = result.replacingOccurrences(of: ".conv_tr.", with: ".conv_tr1d.")
            result = result.replacingOccurrences(of: ".rewrite.", with: ".rewrite1d.")
        }
        
        return result
    }
    
    /// Load weights from a safetensors file
    public static func loadWeights(from path: String) throws -> ModuleParameters {
        let url = URL(fileURLWithPath: path)
        var weights = try MLX.loadArrays(url: url)
        
        // Convert keys from Python to Swift format and filter out unwanted keys
        var convertedWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            // Filter out num_batches_tracked
            if key.contains("num_batches_tracked") {
                continue
            }
            
            let swiftKey = convertPythonKeyToSwift(key)
            
            // Cast to float32 if needed
            if value.dtype != .float32 {
                convertedWeights[swiftKey] = value.asType(.float32)
            } else {
                convertedWeights[swiftKey] = value
            }
        }
        
        return ModuleParameters.unflattened(convertedWeights)
    }
}
