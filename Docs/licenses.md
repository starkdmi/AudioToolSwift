# Licenses and attribution

AudioToolSwift is Apache-2.0. This file records third-party material that ships **in
this repository**, and the licences of model weights fetched at runtime.

> **Status: incomplete.** Only entries verified so far are listed. The full audit -
> covering every model's code licence, weights licence, redistribution status and
> hosting decision - is Phase 4 of the release plan. Do not treat an absence here as
> a clean bill.

## Data included in this repository

### AudioSet ontology — CC BY 4.0

`Sources/AudioToolCore/Types/SoundEmbedding+AudioSet.swift` contains the 527 AudioSet
class names and machine IDs, generated from `class_labels_indices.csv`.

`Sources/AudioToolCore/Types/SoundEmbedding+Presets.swift` groups those class indices
into the seven bundled presets.

- **Source:** AudioSet, Google Research — <https://research.google.com/audioset/>
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Attribution:** Jort F. Gemmeke et al., *"AudioSet: An ontology and human-labeled
  dataset for audio events"*, ICASSP 2017.
- **Modifications:** class names and IDs transcribed into Swift arrays; no changes to
  names, indices, or their ordering.

Only the ontology - the names and their indices - is used. No AudioSet audio or
labelled recordings are included.

### USS conditioning embeddings

`Sources/Models/USS/Embeddings/*.safetensors` — seven 2 KB vectors over the AudioSet
classes above. These are inputs that select what USS extracts, not model weights.
Covered by the USS/ResUNet30 entry below once that audit is complete.

## Model weights

None are committed. Every model downloads its weights at runtime from HuggingFace, so
their licences bind users of this library rather than this repository. Per-model
licence verification is outstanding — see the release plan's Phase 4.

Known so far:

| Model | Upstream | Code | Weights | Verified |
|---|---|---|---|---|
| MossFormer2 SE / SS / SR, FRCRN, MossFormerGAN | ClearerVoice-Studio | Apache-2.0 | Apache-2.0 | ✅ audited |
| RUAccent | Den4ikAI/ruaccent | MIT | MIT | ✅ see note |
| Demucs (HTDemucs) | — | — | — | ❌ |
| USS / ResUNet30 | ByteDance | — | — | ❌ |
| Kokoro | hexgrad/Kokoro-82M | — | — | ❌ |
| Chatterbox | ResembleAI | — | — | ❌ |

**RUAccent note.** Current `main` is MIT (© Denis Petrov), and the HuggingFace model
repos are tagged MIT. PyPI 1.5.8.3, the repo's own `setup.py`/`pyproject.toml`
classifiers, and older HF revisions still say Apache-2.0, so automated scanners will
report Apache. Both permit commercial use; pin the revision the conversion came from
and carry that revision's notice.

**NISQA.** Its evaluation weights are CC BY-NC-SA 4.0. Do not vendor them, ship them,
or use them in the benchmark harness.

## Third-party Swift sources vendored here

| Path | Upstream | Licence |
|---|---|---|
| `Sources/Models/Kokoro/` | mlalma/kokoro-ios | MIT (`LICENSE` carried through) |

## Dependencies fetched by SwiftPM

Resolved as packages, not vendored: `mlx-swift`, `mlx-swift-lm`, `swift-transformers`,
`SwiftAudio`, `MisakiSwift`, `MLXUtilsLibrary`, `FluidAudio`.

FluidAudio additionally downloads diarization, Parakeet ASR and Silero VAD models from
`FluidInference/*` at runtime. Those are not redistributed here; see FluidAudio's own
licensing, and note upstream terms (pyannote's, for instance) may still bind end users.
