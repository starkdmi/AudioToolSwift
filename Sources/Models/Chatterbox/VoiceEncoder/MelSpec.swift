import Foundation
import MLX

public func melspectrogram(_ wav: MLXArray, hp: VoiceEncConfig, pad: Bool = true) -> MLXArray {
    var wav = wav
    let was1d = wav.ndim == 1
    if was1d {
        wav = expandDims(wav, axis: 0)
    }

    var specs: [MLXArray] = []
    specs.reserveCapacity(wav.shape[0])
    for i in 0..<wav.shape[0] {
        let spec = DSP.stft(
            wav[i],
            nFFT: hp.n_fft,
            hopLength: hp.hop_size,
            winLength: hp.win_size,
            window: "hann",
            center: true
        )
        specs.append(spec)
    }

    let spec = stacked(specs, axis: 0)
    var specMag = MLX.abs(spec)
    if hp.mel_power != 1.0 {
        specMag = MLX.pow(specMag, MLXArray(hp.mel_power))
    }

    let filters = DSP.melFilters(
        sampleRate: hp.sample_rate,
        nFFT: hp.n_fft,
        nMels: hp.num_mels,
        fMin: Float(hp.fmin),
        fMax: Float(hp.fmax),
        norm: "slaney",
        melScale: "slaney"
    )

    let filtersT = filters.transposed(0, 1)
    var mel = specMag.matmul(filtersT)
    mel = mel.transposed(0, 2, 1)

    if hp.mel_type == "db" {
        mel = 20.0 * MLX.log10(MLX.maximum(mel, hp.stft_magnitude_min))
    }

    if hp.normalized_mels {
        let minLevelDb = 20.0 * Float(Foundation.log10(Double(hp.stft_magnitude_min)))
        let headroomDb: Float = 15.0
        let minDb = MLXArray(minLevelDb, dtype: mel.dtype)
        let denom = MLXArray(-minLevelDb + headroomDb, dtype: mel.dtype)
        mel = (mel - minDb) / denom
    }

    return was1d ? mel.squeezed(axis: 0) : mel
}
