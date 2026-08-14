# Test fixtures

All WAV files here are redistributable: the speech fixtures are CC0 or
CC BY 4.0, and the music fixture is CC0. See `Docs/licenses.md` for the
attribution each one requires.

## Conversion-validation inputs — VCTK, CC BY 4.0

These three files retain the inputs used while validating the corresponding
Python-to-MLX conversions.

| File | Format | Model/use | Source |
|---|---|---|---|
| `test.wav` | 16 kHz mono | FRCRN SE 16K | resampled VCTK `p232_005.wav` |
| `test_48k.wav` | 48 kHz mono | MossFormer2 SE 48K | VCTK `p232_005.wav` |
| `sr_input_16k.wav` | 16 kHz mono | MossFormer2 SR 48K | resampled VCTK `p232_005.wav` |

VCTK v0.92 is CC BY 4.0. Required attribution and the source record are in
`Docs/licenses.md`.

## CC0 multi-speaker sources

The remaining speech fixtures are derived from three individual English
Voice-Zero samples mirrored in `kyutai/tts-voices`. Kyutai explicitly releases
its `voice-zero/` directory under CC0. The mirror is pinned at revision
`323332d33f997de8394f24a193e1a76df720e01a`.

| Source | SHA-256 |
|---|---|
| `voice-zero/caro_davy.wav` | `40c692c005a0268a7a5b6ebae348077d3dca6a86eb6b12bd36e343bbcd71b5f6` |
| `voice-zero/bill_boerst.wav` | `be4815e4fb760ba1b78117545a260cce4a4c124c7657bc5c6127a0fef8ba661f` |
| `voice-zero/peter_yearsley.wav` | `fbb3920fda7ae26a5a8b317ffcae1d55c0bd5d89d075205f5a52b1e924b83f51` |

Sources:

- <https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/README.md>
- <https://github.com/OwenTyme/voice-zero>

## Derived speech fixtures — CC0

| File | Format | Model/use | Construction | Current SHA-256 |
|---|---|---|---|---|
| `speech_dialogue.wav` | 17.5 s, 16 kHz mono | ASR, VAD, diarization, USS, 2SPK | Caro then Bill, with a clear turn pause | `81bdb40958f9227a695cb0cf861c1b1c387fa5374e72f6ea344f3157a0e0571f` |
| `mix_8k.wav` | 8.25 s, 8 kHz mono | MossFormer2 2SPK / WHAMR | Caro and Bill overlapping; Bill delayed 250 ms | `10937c604e3a4fc442df4e3cd608c342abe07fe71226d50edd338d1160b0347e` |
| `mix_16k.wav` | 8.25 s, 16 kHz mono | MossFormer2 2SPK 16K parity | the `mix_8k` recipe at 16 kHz | `e4c24eeb288c60ff491ab13752db9f9d3d0902739d44448670f5256a225ff2fa` |
| `mix3_8k.wav` | 8.25 s, 8 kHz mono | MossFormer2 3SPK | Caro, Bill and Peter overlapping; staggered starts | `bfe04667b38a1c476548383f12f441a898736d94e5eff3ef3c047c4ee4032505` |
| `multi_speaker.wav` | 17.5 s, 16 kHz stereo | stereo VAD / multi-speaker loading | spatialised Caro/Bill dialogue | `a7d47fcc1b4b84e27b525ee8a8076204f0d52ff6963118722932b1943c2b6d82` |
| `speech_long.wav` | 25.5 s, 22.05 kHz mono | streaming and pipeline integration | sequential Caro, Bill and Peter with short overlaps | `1ed049cc03de09774a552ee1426b1b76b065f248399690675d9f45d2cfa44586` |
| `speech_music.wav` | 10 s, 44.1 kHz stereo | VAD over music / Demucs vocals | Caro mixed over “Distant Wonders” | `0d3e6a07c0755564829c214c04697c47f13e83dffa0556256e91daededb197d6` |
| `speech_music_32k.wav` | 10 s, 32 kHz mono | USS conversion validation | `speech_music.wav` at USS's own rate | `8f8df2c10b943ab3fd0d1953f9ac0596d824e8b2b289be8897fc86d26429e910` |
| `speech_24k.wav` | 3 s, 24 kHz mono | Chatterbox conversion validation | Caro at S3Gen's own rate | `60c9fa9b39679b3ef8285dc45a999e23e520a6b5013994e864afef57135e28c4` |

Gain changes, fades, delays, panning, resampling and PCM encoding are specified
in `Scripts/fetch-fixtures.sh`. The new separation mixtures intentionally are
not byte-equivalent to the old conversion-validation inputs.

The last three were also used for cross-language conversion validation. They give
the 2SPK 16K, USS and Chatterbox checks a redistributable input at each model's
own rate.

## Music fixture — CC0

`music.wav` is a ten-second 44.1 kHz stereo excerpt of “Distant Wonders” from
*CC0 Instruments, Volume I*:
<https://archive.org/details/MarchForHonor>.
The downloaded `Distant_Wonders.mp3` source has SHA-256
`cc56ebba3109a6c18c18c7375c88fa2043769ddacc52360a22602fc9d5fb4675`;
the derived `music.wav` has SHA-256
`4f925ce9040f7586829357c956bb3b1386ef19dd94fd6282f296384f8e0cb337`.

`speech_music.wav` mixes the CC0 Caro voice over that excerpt. It provides a
speech-over-music case while leaving the pure instrumental input available for
Demucs parity.

## Reproduction

Run `Scripts/fetch-fixtures.sh`. It downloads only the three individual speech
samples above plus the CC0 music track; it does not download any complete
dataset archive. Source hashes are verified before derivation, and two
consecutive runs produce identical bytes for every file in the table.

That last part had to be fixed: `mix_8k` and `mix3_8k` were built with ffmpeg's
`amix=normalize=1`, which divides by the number of inputs that happen to be
active, so each run produced one of two files differing over the final quarter
second. Both now use `normalize=0` with an explicit 1/N gain. Any hash for those
two recorded before 2026-08-10 is one of the two coin-flip results.

Coverage: 8 kHz covers the 3-speaker and WHAMR variants; 16 kHz covers FRCRN,
MossFormer2 SS, MossFormerGAN, ASR, VAD and diarization; 22.05 kHz exercises
resampling/streaming; 24 kHz covers Chatterbox conditioning; 32 kHz covers USS;
44.1 kHz stereo covers Demucs and mixed-media VAD; and 48 kHz covers
MossFormer2 SE.
