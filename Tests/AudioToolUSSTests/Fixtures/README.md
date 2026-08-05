# Test fixtures

Audio fixtures are **not** committed. This directory exists so SwiftPM declares a
resource bundle for the target, which is what makes `Bundle.module` available; the
integration tests that read these files are gated behind
`TestConfig.shouldRunIntegrationTests` and skip when the audio is absent.

## Why they are not in the repo

The original fixtures were third-party recordings — an audiobook, commercial music,
broadcast audio — that we do not own and cannot redistribute. Their processed
derivatives (separated stems, enhanced output) inherit that status, so those cannot
be published either. They were removed from the entire git history rather than just
the working tree.

## Expected files

| File | Used by |
|---|---|
| `harry_potter.wav` | transcription, diarization, word timing |
| `watson_30s.wav` | diarization, speaker separation |
| `music_35s.wav` | Demucs / music separation |
| `billions.wav` | multi-speaker diarization |
| `test.wav` | enhancement smoke tests |
| `mix_8k.wav`, `mix3_8k.wav` | MossFormer2 SS 2-speaker and 3-speaker |

## Obtaining them

Two tiers, deliberately kept apart:

- **Private regression set** — the original recordings, for confirming that a change
  did not alter output. Never referenced by CI and never published.
- **Public set** — openly licensed replacements used by CI, the benchmark harness and
  the comparison page. Sourced from VoiceBank-DEMAND: VCTK (CC BY 4.0) for speech and
  DEMAND (CC BY-SA 3.0) for noise. That pairing is also the standard evaluation set
  for MossFormer2 SE and FRCRN, so measurements taken against it are directly
  comparable to the published papers. Music separation uses Free Music Archive
  CC-BY tracks rather than MUSDB18, which is partly non-commercial.

Drop the files into this directory to run the integration tests locally.
