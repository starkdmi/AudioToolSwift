# Test fixtures

Small, openly licensed audio, committed to the repository. Regenerate with
`Scripts/fetch-fixtures.sh`, which reproduces every file here byte-for-byte.

## Provenance

| File | Rate | Source | Licence |
|---|---|---|---|
| `speech_long.wav` | 22.05 kHz | ClearerVoice-Studio `samples/path_to_input_wavs_sr/LJ001-0001.wav` (LJSpeech) | Apache-2.0 |
| `speech_dialogue.wav` | 16 kHz | ClearerVoice-Studio `samples/input_ss.wav` | Apache-2.0 |
| `multi_speaker.wav` | 16 kHz stereo | ClearerVoice-Studio `samples/path_to_input_wavs_ss/speech_mixure1.wav` | Apache-2.0 |
| `test.wav` | 16 kHz | derived: three ClearerVoice-Studio speech samples concatenated | Apache-2.0 |
| `test_48k.wav` | 48 kHz | ClearerVoice-Studio `samples/test.wav` | Apache-2.0 |
| `speech_48k.wav` | 48 kHz | derived: `speech_long` resampled | Apache-2.0 |
| `mix_8k.wav` | 8 kHz | derived: ClearerVoice-Studio 2-speaker mixture, resampled | Apache-2.0 |
| `mix3_8k.wav` | 8 kHz | derived: three ClearerVoice-Studio talkers overlapped | Apache-2.0 |
| `music.wav` | 44.1 kHz stereo | *Distant Wonders*, "CC0 Instruments, Volume I", archive.org | CC0 1.0 |

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) is the upstream
of MossFormer2 SE/SS/SR and FRCRN, so these are the clips those models are demoed on -
which makes them the most defensible inputs available for benchmarking them, and they
carry the same Apache-2.0 licence as this project.

## Why the previous fixtures are gone

They were third-party recordings - an audiobook, commercial music, broadcast audio -
that we did not own and could not redistribute, along with derived outputs that
inherited the same status. They were removed from the entire git history, not just
the working tree.

## Coverage

`test.wav` is deliberately built rather than copied: it backs the VAD, transcription
and diarization tests, which need sustained 16 kHz speech, and ClearerVoice's own
`test.wav` is 48 kHz and not speech-dominant enough for VAD to segment.

16 kHz for FRCRN, MossFormer2 SS and MossFormerGAN; 8 kHz for the 3-speaker and WHAMR
separation variants; 48 kHz for MossFormer2 SE 48K and super-resolution output;
22.05 kHz as super-resolution input; 44.1 kHz stereo for Demucs.

Not yet covered: varied sound types for USS - it needs material spanning several
AudioSet categories (animal, nature, things) rather than speech and music alone.
