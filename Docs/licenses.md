# Third-party licences and provenance

AudioToolSwift's own code is Apache-2.0. This document records the third-party
material that ships in this repository, the model implementations vendored into
it, and the weights or assets that its APIs download at runtime.

It reflects the licences and provenance published by upstream maintainers as of
2026-08-10. It is an engineering provenance record, not legal advice.

## Data included in this repository

### AudioSet class labels — CC BY 4.0

`Sources/AudioToolCore/Types/SoundEmbedding+AudioSet.swift` contains the 527
AudioSet display names and machine IDs transcribed from
`class_labels_indices.csv`. `SoundEmbedding+Presets.swift` groups indices into
seven package presets.

- **Source:** [Google Research AudioSet](https://research.google.com/audioset/download.html),
  `class_labels_indices.csv`
- **Licence:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- **Attribution:** Jort F. Gemmeke et al., “AudioSet: An ontology and
  human-labeled dataset for audio events,” ICASSP 2017.
- **Modifications:** display names and IDs were transcribed into Swift arrays;
  package-specific preset groupings were added.

Google licenses the AudioSet **dataset** under CC BY 4.0 and the **ontology**
under CC BY-SA 4.0, and its download page draws that line itself.
`class_labels_indices.csv` is distributed with the dataset, and it is the only
AudioSet file this package was built from — `ontology.json` was not used. What is
reproduced here is the label set the model's 527-way output vector is indexed by:
names, machine IDs and order. None of the ontology's hierarchy, descriptions,
positive-example references or restriction flags appears.

So the obligations that attach are CC BY 4.0's — attribution and an indication of
changes, both given above — and not ShareAlike. The generated arrays are not
Apache-2.0 material either way, and the attribution stands whichever of the two
licences a reader thinks applies; the difference is only whether ShareAlike
reaches them, and on the reasoning above it does not. This is an engineering
provenance record, not legal advice.

### USS conditioning embeddings

USS is conditioned on a 527-dimensional vector over the AudioSet classes, which
selects what the separator extracts. It is an input, not a model weight.

Seven of these shipped as `Sources/Models/USS/Embeddings/*.safetensors`, ~2 KB
each. They are gone, and the code reproduces them exactly. The generation rule is
USS's own `at_soft` conditioning, and it is complete in one line:

> a 527-dimensional zero vector, 1.0 at each AudioSet index in the group,
> divided by the number of indices in the group.

So each vector holds two distinct values, `0` and `1/n`, and is fully described
by its class list. Those lists are in `SoundEmbedding+Presets.swift`, and
`SoundEmbeddingParityTests` pins each reconstructed preset to the SHA-256 of the
original conversion's float32 payload.

No audio and no audio-tagging model is involved in producing them — they are
hand-authored groupings, not inference output. The only upstream material they
derive from is the AudioSet class list covered above.

## Test fixtures

Every WAV file in this repository is redistributable: the speech fixtures are
CC0 or CC BY 4.0 and the music fixture is CC0. `Scripts/fetch-fixtures.sh`
downloads the pinned CC0 sources, verifies every source hash, and reproduces
each derived fixture byte for byte. It fetches individual small files, not
corpus archives. Per-file formats, constructions and hashes are in the fixture
READMEs under `Tests/**/Fixtures/`.

| Tracked file(s) | Identified source | Terms |
|---|---|---|
| `test_48k.wav` | VCTK `p232_005.wav` | CC BY 4.0 |
| `test.wav` and `sr_input_16k.wav` | 16 kHz resamples of VCTK `p232_005.wav` | CC BY 4.0; resampling identified below |
| `music.wav` | excerpt derived from “Distant Wonders,” *CC0 Instruments, Volume I* | CC0 1.0 |
| Both copies of `speech_dialogue.wav` | mostly sequential mix of Voice-Zero `caro_davy.wav` and `bill_boerst.wav`, resampled to 16 kHz mono | CC0; mixing, gain, fades and resampling documented |
| `mix_8k.wav` | overlapping mix of `caro_davy.wav` and `bill_boerst.wav`, resampled to 8 kHz mono | CC0; mixing, gain, fades and resampling documented |
| `mix_16k.wav` | the same mixture at 16 kHz, for the 2SPK 16K parity case | CC0; mixing, gain, fades and resampling documented |
| `mix3_8k.wav` | overlapping mix of `caro_davy.wav`, `bill_boerst.wav` and `peter_yearsley.wav`, resampled to 8 kHz mono | CC0; mixing, gain, fades and resampling documented |
| `multi_speaker.wav` | spatialised stereo dialogue derived from `caro_davy.wav` and `bill_boerst.wav`, resampled to 16 kHz | CC0; mixing, gain, fades, panning and resampling documented |
| `speech_long.wav` | 25.5-second sequence derived from all three Voice-Zero samples, resampled to 22.05 kHz mono | CC0; sequencing, gain, fades and resampling documented |
| `speech_music.wav` | Voice-Zero `caro_davy.wav` mixed over the “Distant Wonders” excerpt at 44.1 kHz stereo | Both inputs CC0; mixing, gain, fades and resampling documented |
| `speech_music_32k.wav` | `speech_music.wav` at 32 kHz mono, for the USS parity case | Both inputs CC0; resampling documented |
| `speech_24k.wav` | Voice-Zero `caro_davy.wav`, first 3 s at 24 kHz mono, for the Chatterbox parity case | CC0; gain, fades and resampling documented |

The multi-speaker fixtures are deterministic mixtures of three small English
Voice-Zero samples from the pinned Kyutai mirror revision
`323332d33f997de8394f24a193e1a76df720e01a`. Kyutai identifies its `voice-zero/`
directory as CC0.

| Source file | SHA-256 |
|---|---|
| `caro_davy.wav` | `40c692c005a0268a7a5b6ebae348077d3dca6a86eb6b12bd36e343bbcd71b5f6` |
| `bill_boerst.wav` | `be4815e4fb760ba1b78117545a260cce4a4c124c7657bc5c6127a0fef8ba661f` |
| `peter_yearsley.wav` | `fbb3920fda7ae26a5a8b317ffcae1d55c0bd5d89d075205f5a52b1e924b83f51` |
| `Distant_Wonders.mp3` | `cc56ebba3109a6c18c18c7375c88fa2043769ddacc52360a22602fc9d5fb4675` |

Required VCTK attribution:

> Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald (2019), CSTR VCTK
> Corpus version 0.92, University of Edinburgh, DOI 10.7488/ds/2645,
> licensed CC BY 4.0. AudioToolSwift resampled `p232_005.wav` to produce the
> 16 kHz derivatives.

Primary fixture sources:

- [VCTK v0.92 record and authors](https://datashare.ed.ac.uk/handle/10283/3443)
  and [licence text](https://datashare.ed.ac.uk/server/api/core/bitstreams/956a1688-0b59-428c-8a2f-10837433dde3/content)
- [Pinned Kyutai TTS voices provenance](https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/README.md)
  and [Voice-Zero source project](https://github.com/OwenTyme/voice-zero)
- [Internet Archive item metadata for CC0 Instruments](https://archive.org/details/MarchForHonor)

### Parity artifacts

`Parity/artifacts/` is generated, gitignored, and intended for publication
alongside the weights (`Parity/README.md`). An artifact stores its **input
samples**, not a path to a wav, so it carries a copy of whatever audio the case
was built from. Every case therefore reads a committed fixture from the table
above, and each sidecar records its source path and SHA-256.

## Model weights and runtime assets

No model weights are tracked in this repository. The package does name and
download default repositories, so a host application must surface and comply
with the applicable model terms. A runtime download does not turn a third-party
model into Apache-2.0 material.

**Every repository below is pinned.** `ModelPins` records a commit and the
SHA-256 of each file for every configured repository, and `ModelDownloader`
resolves each download against that revision and verifies the bytes it wrote
before returning a path. A host therefore gets the same weights, under the same
published terms, as the ones audited here — an upstream change to a default
branch cannot silently alter either. The exceptions are the ASR, diarization and
Sortformer models, which FluidAudio downloads itself and which are pinned by its
package version rather than by this table.

No configured model carries a non-commercial clause. MIT, Apache-2.0 and CC BY
4.0 all permit commercial use, and NVIDIA's Open Model License explicitly says
its models are commercially usable. TranslateGemma is governed by the Gemma
Terms and prohibited-use policy rather than an open-source licence; those terms
are not a blanket non-commercial licence, but they do bind distribution.

| Feature / configured repository | Effective terms | What a host must carry |
|---|---|---|
| MossFormer2 SE, SR, 2-speaker and 3-speaker conversions under `starkdmi/MossFormer2_*_MLX` | Apache-2.0, from ClearerVoice code and the Alibaba model cards | Apache attribution and NOTICE material |
| `starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX` | MIT, from base `alibabasglab/mossformer2-whamr-2spk` | The upstream MIT notice |
| FRCRN `starkdmi/FRCRN_SE_16K_MLX` | Apache-2.0, from base `alibabasglab/FRCRN_SE_16K` | Apache attribution and NOTICE material |
| USS `starkdmi/USS_MLX` | Apache-2.0, from ByteDance USS code and `RSNuts/Universal_Source_Separation` weights | Apache attribution and NOTICE material |
| Demucs v4 / HTDemucs-ft `starkdmi/Demucs_MLX` | MIT, for both `adefossez/demucs` code and the official HTDemucs-ft weights | The MIT notice. Official source-specific checkpoints: `f7e0c4bc` (drums), `d12395a8` (bass), `92cfc3b6` (other), `04573f0d` (vocals) |
| Kokoro `mlx-community/Kokoro-82M-{bf16,4bit,6bit,8bit}` | Apache-2.0, from `hexgrad/Kokoro-82M` and the inspected MLX variants | Model attribution, plus `Sources/Models/Kokoro/LICENSE` for the vendored port |
| Chatterbox `starkdmi/chatterbox` and precision-suffixed variants | MIT, from the base ResembleAI checkpoint and code | The Resemble AI MIT notice |
| Chatterbox S3 tokenizer fallback `mlx-community/S3TokenizerV2` | Apache-2.0, via `FunAudioLLM/CosyVoice2-0.5B` | The CosyVoice2 upstream notice |
| Whisper Large v3 `mlx-community/whisper-large-v3-mlx` | MIT, from OpenAI Whisper | The OpenAI MIT notice |
| Whisper Small `mlx-community/whisper-small-mlx` | MIT, from OpenAI Whisper | The OpenAI MIT notice |
| Silero VAD, via FluidAudio `FluidInference/silero-vad-coreml` | MIT | The upstream MIT notice |
| FluidAudio Parakeet v2/v3 Core ML | CC BY 4.0, from the NVIDIA base models | Model name, creator, source, licence link, and a conversion/modification notice. The v3 card body says Apache-2.0, but its metadata and base model say CC BY 4.0; CC BY 4.0 controls |
| FluidAudio offline speaker diarization | CC BY 4.0, from pyannote `speaker-diarization-community-1` | Attribution and modification notice. Upstream pyannote also gates initial access on accepting contact terms |
| FluidAudio streaming Sortformer | NVIDIA Open Model License, from the NVIDIA base model | A copy of the agreement and a NOTICE containing “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” |
| TranslateGemma `mlx-community/translategemma-4b-it-4bit` | Gemma Terms of Use | Downstream restrictions, a copy of the terms, modification notices, and the prescribed NOTICE outside a hosted service. Surface acceptance in the host; do not present it as an Apache model |
| RUAccent caller-supplied assets | MIT in the current GitHub/Hugging Face release; older PyPI metadata and revisions say Apache-2.0 | Neither downloaded nor bundled by this package. The host pins and documents the exact local model/dictionary revision and retains that revision's notice |
| MossFormerGAN Core ML `starkdmi/MossFormer_GAN_SE_16K_CoreML` | Apache-2.0, from base `alibabasglab/MossFormerGAN_SE_16K` | Apache attribution and NOTICE material |

Primary model sources:

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio),
  [MossFormer2 SE](https://huggingface.co/alibabasglab/MossFormer2_SE_48K),
  [MossFormer2 SS](https://huggingface.co/alibabasglab/MossFormer2_SS_16K),
  [WHAMR base](https://huggingface.co/alibabasglab/mossformer2-whamr-2spk),
  [FRCRN](https://huggingface.co/alibabasglab/FRCRN_SE_16K),
  and [MossFormerGAN SE](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)
- [ByteDance USS](https://github.com/bytedance/uss) and
  [published USS checkpoint](https://huggingface.co/RSNuts/Universal_Source_Separation)
- [Maintained Demucs v4 source](https://github.com/adefossez/demucs),
  [standard HTDemucs weights](https://huggingface.co/adefossez/HTDemucs), and
  [four-model HTDemucs-ft weights](https://huggingface.co/adefossez/HTDemucs-ft)
- [Kokoro base](https://huggingface.co/hexgrad/Kokoro-82M) and
  [default MLX conversion](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Chatterbox base model](https://huggingface.co/ResembleAI/chatterbox),
  [Resemble AI MIT licence](https://github.com/resemble-ai/chatterbox/blob/master/LICENSE),
  [configured conversion](https://huggingface.co/starkdmi/chatterbox),
  [S3TokenizerV2](https://huggingface.co/mlx-community/S3TokenizerV2), and
  [CosyVoice2 base](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)
- [OpenAI Whisper licence](https://github.com/openai/whisper/blob/main/LICENSE),
  [Whisper Large v3 MLX](https://huggingface.co/mlx-community/whisper-large-v3-mlx),
  and [Whisper Small MLX](https://huggingface.co/mlx-community/whisper-small-mlx)
- [Silero VAD Core ML](https://huggingface.co/FluidInference/silero-vad-coreml),
  [Parakeet v2 Core ML](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml),
  [Parakeet v3 Core ML](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml),
  [speaker diarization Core ML](https://huggingface.co/FluidInference/speaker-diarization-coreml),
  and [Sortformer Core ML](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml)
- [NVIDIA Sortformer base](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
  and [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- [TranslateGemma MLX](https://huggingface.co/mlx-community/translategemma-4b-it-4bit)
  and [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- [RUAccent](https://github.com/Den4ikAI/ruaccent)

## Third-party Swift sources vendored here

These directories are compiled as AudioToolSwift targets rather than resolved as
separate packages.

| Path | Principal upstream | Licence |
|---|---|---|
| `Sources/Models/MossFormer2SE/` | ClearerVoice / local MLX port | Apache-2.0 |
| `Sources/Models/MossFormer2SS/` | ClearerVoice / local MLX port | Apache-2.0; WHAMR weights separately MIT |
| `Sources/Models/MossFormer2SR/` | ClearerVoice / local MLX port | Apache-2.0 |
| `Sources/Models/FRCRN/` | ClearerVoice / local MLX port | Apache-2.0 |
| `Sources/Models/USS/` | ByteDance USS / local MLX port | Apache-2.0 |
| `Sources/Models/Demucs/` | adefossez/demucs HTDemucs-ft / local MLX port | MIT for code and official weights |
| `Sources/Models/Kokoro/` | mlalma/kokoro-ios | MIT, notice at `Sources/Models/Kokoro/LICENSE` |
| `Sources/Models/Chatterbox/` | Resemble AI Chatterbox / local MLX port | MIT |

Individual files also record algorithms adapted from other ports. Any
consolidated notice must preserve every applicable upstream notice rather than
assigning one blanket licence to a mixed-provenance directory.

## SwiftPM dependencies

The exact revisions are recorded in `Package.resolved`. Licences below were
verified from the corresponding resolved checkout:

| Resolved package | Version/revision | Licence |
|---|---|---|
| `mlx-swift` | 0.29.1 | MIT |
| `mlx-swift-lm` | 2.29.3 | MIT |
| `swift-transformers` | 1.1.9 | Apache-2.0 |
| `SwiftAudio` | 1.2.2 | Apache-2.0 |
| `MisakiSwift` fork | `1ecaf9a6057ed8bdd69e5a37cdcc0b5cb30eb901` | Apache-2.0 |
| `MLXUtilsLibrary` | 0.0.6 | Apache-2.0 |
| `FluidAudio` | `5390df9752c8fc583596018360c5fd70d6fa6c75` | Apache-2.0 |
| Apple `swift-asn1`, `swift-collections`, `swift-crypto`, `swift-numerics` | see `Package.resolved` | Apache-2.0 |
| `swift-jinja` | 2.4.2 | Apache-2.0 |
| `ZIPFoundation` | 0.9.20 | MIT |
| `yyjson` | 0.12.0 | MIT |

SwiftPM retains source checkout licences for developers, but an application
distributed in binary form must still carry the notices those licences require.
