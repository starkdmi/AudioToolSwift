# Test fixtures

These files are copies shared with the FluidAudio integration tests. Both
permit redistribution; see `Docs/licenses.md` and the FluidAudio fixture README
for full provenance and hashes.

| File | Source | Terms |
|---|---|---|
| `test.wav` | 16 kHz resample of VCTK `p232_005.wav` | CC BY 4.0; attribution and resampling notice required |
| `speech_dialogue.wav` | dialogue derived from Voice-Zero `caro_davy.wav` and `bill_boerst.wav` | CC0 |

`Scripts/fetch-fixtures.sh` regenerates `speech_dialogue.wav` from two pinned,
hash-verified source clips and copies it into this directory.
