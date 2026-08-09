//
//  ParityThresholds.swift
//  AudioToolParityTests
//
//  The recorded floor for each comparison, in dB.
//

import Foundation

/// Minimum acceptable SNR against the MLX Python reference, per tensor.
///
/// **Derived, not invented.** Each entry is a number that was measured first with
/// `PARITY_RECORD=1` and then written down with margin, and the comment on each
/// says what was observed. A threshold chosen before measuring only encodes what
/// someone hoped for.
///
/// For scale: float32 round-off alone lands around 110-130 dB, and
/// `Models/python/frcrn_se_mlx/README.md` records 115.98 dB for MLX against
/// PyTorch. Anything under roughly 60 dB is a real difference in the port, not
/// accumulated error.
///
/// The 23.8 dB in the CoreML GAN README is *not* a threshold for anything here.
/// That figure is CoreML against PyTorch - conversion loss, already paid for when
/// the 256-frame variant was chosen. Both sides of the GAN case run the same
/// `.mlpackage` through the same MLX STFT, so its floor belongs with the rest.
enum ParityThresholds {

    /// Keyed by `"<case>.<tensor>"`.
    ///
    /// A missing entry fails loudly with the measured value rather than defaulting
    /// to something permissive - a parity suite that silently passes unmeasured
    /// cases is worse than no suite. Cases still under investigation are listed at
    /// the bottom and deliberately left out.
    static let minimumSNR: [String: Double] = [
        // USS. Measured 102.5 and 131.1 dB - the port is float32-exact.
        "uss_resunet30_32k.separated_speech": 90,
        "uss_resunet30_32k.separated_music": 110,

        // Demucs, per-stem checkpoints, stereo in and mono out. Measured 91.3,
        // 86.9, 74.7 and 132.3 dB.
        //
        // vocals is the lowest of the four and it is also the quietest stem on
        // this clip - music.wav is instrumental, so vocals sits near 1e-4 RMS and
        // round-off is a larger share of a smaller signal. Not a weaker port; a
        // weaker fixture. See the note in Parity/adapters/demucs.py.
        "demucs_vocals_44k.drums_mono": 80,
        "demucs_vocals_44k.bass_mono": 75,
        "demucs_vocals_44k.vocals_mono": 65,
        "demucs_vocals_44k.other_mono": 110,

        // MossFormerGAN, CoreML. Measured 129.0 dB on one exact segment and 129.6 dB
        // on the whole clip through the segmenting path - so both the model plus
        // STFT and the stitching around it agree to round-off.
        //
        // These read -1.1 and -1.2 dB before parseModelOutputToMLX was taught to
        // honour the output array's strides. That is the single largest thing this
        // harness has found: a provider that had never produced correct output, in a
        // way no amount of listening to the Python side would have surfaced.
        "mossformer_gan_se_16k_coreml.enhanced_segment": 110,
        "mossformer_gan_se_16k_coreml.enhanced_full": 110,

        // FRCRN, chunked at 4s/25%/discard-edges. Measured 119.8 dB.
        //
        // Read 41.1 dB against a whole-file reference, with 65% of the error in the
        // last 2400 samples. FRCRN's provider has no direct path - it chunks
        // everything - so an unchunked reference was never a fair comparison.
        "frcrn_se_16k.enhanced": 100,

        // MossFormer2 SE. Measured 80.7 dB direct, 67.1 dB chunked.
        //
        // The lowest pair of the round-off group, and the gap between the two is the
        // cost of the seams. Worth revisiting if it drifts further, but 67 dB is
        // ~2000x below the signal.
        "mossformer2_se_48k.enhanced": 55,
        "mossformer2_se_48k_direct.enhanced": 70,

        // MossFormer2 SS, all three configurations, chunked at 4s/25%/triangular.
        // Measured 91.5 to 107.5 dB. Previously 11.7 to 36.2 dB against whole-file
        // references - every one of those was the harness, not the port.
        "mossformer2_ss_2spk_16k.speaker_1_normalized": 85,
        "mossformer2_ss_2spk_16k.speaker_2_normalized": 85,
        "mossformer2_ss_2spk_16k_direct.speaker_1_normalized": 85,
        "mossformer2_ss_2spk_16k_direct.speaker_2_normalized": 85,
        "mossformer2_ss_3spk_8k.speaker_1_normalized": 80,
        "mossformer2_ss_3spk_8k.speaker_2_normalized": 80,
        "mossformer2_ss_3spk_8k.speaker_3_normalized": 80,
        "mossformer2_ss_3spk_8k_direct.speaker_1_normalized": 85,
        "mossformer2_ss_3spk_8k_direct.speaker_2_normalized": 85,
        "mossformer2_ss_3spk_8k_direct.speaker_3_normalized": 85,
        "mossformer2_ss_2spk_whamr_8k.speaker_1_normalized": 90,
        "mossformer2_ss_2spk_whamr_8k.speaker_2_normalized": 90,
        "mossformer2_ss_2spk_whamr_8k_direct.speaker_1_normalized": 90,
        "mossformer2_ss_2spk_whamr_8k_direct.speaker_2_normalized": 90,

        // MossFormer2 SR, 16 -> 48 kHz. Measured 110.1 dB chunked, 114.2 dB direct.
        //
        // The longest-running case here, and it took three separate defects:
        //
        //   1. `resampleTo48k` turned 48000 samples into 141312 - a temp-wav round
        //      trip, the loader's default quality instead of Mastering/.max, and an
        //      undrained AVAudioConverter dropping 2688 samples per call. 18.7 dB.
        //      Now SuperResolutionResampler; see testSuperResolutionResamplerMatchesReference.
        //   2. `smoothTransition` built its crossfade with `MLX.linspace(0, 1, ...)`,
        //      whose untyped literals infer Int, so the 100 ms ramp was int64: 4799
        //      zeros and a single 1. The crossfade was a hard cut and the fade region
        //      came out unenhanced. 50.3 dB. Fixed in SwiftAudio 1.1.0.
        //   3. With the direct path then at 114.2 dB, chunked sat at 55.3 - all of it
        //      in the first 12000 samples, and samples 1...7 were exactly zero.
        //      `hannWindow` used `0.5 * (1 - cos(2πi/(N-1)))`, and in Float32 `cos(x)`
        //      rounds to 1 for small x, so the subtraction cancelled to zero over the
        //      head of a 192000-sample window. Overlap-add then divided by a zero
        //      weight and left silence. `np.hanning` evaluates in Float64 and does
        //      not show it. Now `sin²(πi/(N-1))`, which is exact in Float32.
        //
        // Each one masked the next: (2) held both paths at ~50 dB, which hid (3)
        // entirely, because chunked and direct agreed to within 0.3 dB while both
        // were wrong.
        "mossformer2_sr_48k.enhanced": 95,
        "mossformer2_sr_48k_direct.enhanced": 100,

        // Chatterbox conditioning, two cases. Every token tensor is bit-identical and
        // the continuous ones measure 116.3/131.9/104.8 dB short and 120.5/134.7/109.4
        // long - float32 round-off throughout. Exact equality is the only result worth
        // accepting for codebook indices, so the token floors sit at 120: one flipped
        // token reads about 19 dB and fails loudly.
        //
        // The short case was 43.6 dB and 11 of 75 tokens wrong before both sides were
        // put on one specified resampler. See the note below, kept because the
        // reasoning is not recoverable from the code.
        //
        // The long case is the one with teeth: 12 s at 22050 Hz, past both the 6 s and
        // 10 s conditioning windows, so the truncations run and the token tensors are
        // 150 and 250 rather than the same 75 twice. 22050 against 24000 also puts the
        // resampler at 160/147 where the short case gets 1/1.
        "chatterbox_conditionals_24k.ve_speaker_emb": 100,
        "chatterbox_conditionals_24k.t3_cond_prompt_tokens": 120,
        "chatterbox_conditionals_24k.s3gen_prompt_token": 120,
        "chatterbox_conditionals_24k.s3gen_prompt_feat": 110,
        "chatterbox_conditionals_24k.s3gen_embedding": 90,

        "chatterbox_conditionals_22k_long.ve_speaker_emb": 100,
        "chatterbox_conditionals_22k_long.t3_cond_prompt_tokens": 120,
        "chatterbox_conditionals_22k_long.s3gen_prompt_token": 120,
        "chatterbox_conditionals_22k_long.s3gen_prompt_feat": 110,
        "chatterbox_conditionals_22k_long.s3gen_embedding": 90,
    ]

    // MARK: - Chatterbox: why the resampler is what it is
    //
    // Resolved, and recorded because the next person to read
    // `Sources/Models/Chatterbox/Utils/Audio.swift` will find two resamplers and no
    // reason for it in the code.
    //
    // `Models/python/chatterbox_mlx` has two different functions both named
    // `resample_audio`: model.py's called librosa, s3gen.py's calls
    // `scipy.signal.resample_poly`. Swift used the scipy-equivalent for both. That
    // read 43.6 dB on `ve_speaker_emb` and 19.4 dB on `s3gen_prompt_token`, with the
    // two tensors whose inputs never leave 24 kHz already at round-off - the split in
    // the numbers was exactly the split in the resamplers.
    //
    // scipy's defaults are the defect, not the port. `resampleAudioPolyphase` is a
    // faithful Double-precision port of `resample_poly(padtype: "edge")`; that filter
    // is 61 taps with its cutoff at Nyquist, where a lowpass is only -6 dB, so
    // content above 8 kHz folds back into the band. On a sweep from 8.2 to 11.8 kHz
    // resampled 24 -> 16, alias rejection measures:
    //
    //     soxr HQ                          -57.7 dB
    //     resampy kaiser_best (published)  -55.8 dB
    //     scipy resample_poly defaults     -20.9 dB
    //
    // The S3 tokenizer resolves that 35 dB into different speech tokens. It is not
    // otherwise twitchy: soxr's HQ, VHQ and MQ tiers sit 55-57 dB apart and give
    // byte-identical tokens, as does additive noise at -80 dBFS.
    //
    // The fix is not to match soxr. Two reasons, and the second is the interesting
    // one. soxr is LGPL-2.1-or-later, so it cannot ship inside an Apache package
    // except as a dynamic dependency. And matching its *output* by fitting a filter
    // to it produces a filter that tracks soxr and is not otherwise good: searching
    // beta and cutoff against these fixtures reaches 1.04% token disagreement, while
    // resampy's published kaiser_best - a better filter than scipy's by 35 dB of
    // alias rejection - disagrees with soxr on 12.84%. Token agreement ranks
    // similarity to a reference implementation, not quality, and optimising it is
    // fitting. That measurement is the reason this file quotes constants instead.
    //
    // So both sides run one published design: resampy's kaiser_best, ISC licensed,
    // beta 12.9846, roll 0.917347, 50 zeros, -120 dB. soxr's own header specifies HQ
    // as passband_end 0.913 - two independent designs on the same passband edge, and
    // Kaiser's formulas (Kaiser 1974; Oppenheim & Schafer 7.5) fix beta and length
    // from the specification rather than from a search. Swift uses it at the four
    // model.py call sites; `S3Gen.embed_ref` stays on scipy's filter because the
    // reference uses scipy there and reproducing it is the requirement. The parity
    // adapter patches the same substitution into the frozen reference tree, and the
    // artifact sidecar records it under `resampler_16k`.
    //
    // What this costs: the artifact no longer reproduces upstream chatterbox, which
    // resamples with librosa. That is a deliberate trade of fidelity-to-upstream for
    // a design that is specified, permissively licensed and identical on both sides.
    // Revisit it if upstream fidelity ever becomes the requirement.
}
