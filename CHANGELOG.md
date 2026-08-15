# Changelog

Notable changes per release. Dates are the tag date; unreleased work sits under
the top heading until it ships.

## 0.1.0 — unreleased

First public release. Everything below describes the state at the tag rather
than a diff against a predecessor, except where it names something that changed
during the release preparation and would surprise a reader of an earlier clone.

### Models

- Speech enhancement: MossFormer2 SE (16 kHz and 48 kHz) and FRCRN over MLX,
  MossFormerGAN over Core ML.
- Speaker separation: MossFormer2 SS, 2-speaker/16 kHz, 3-speaker/8 kHz and the
  WHAMR-trained 2-speaker/8 kHz.
- Super-resolution: MossFormer2 SR, 16 kHz to 48 kHz.
- Source separation: Demucs (vocals or four stems) and USS ResUNet30, the latter
  targetable at any 527-dimension AudioSet class vector.
- Text to speech: Kokoro and Chatterbox over MLX, plus AVSpeechSynthesizer.
- Transcription, VAD and diarization through FluidAudio, and Apple
  SpeechAnalyzer on 26+.
- Translation: Apple's Translation framework and TranslateGemma over MLX.

Weights are fetched from Hugging Face on first use and never committed. Every
repository this package manages is pinned to a commit and verified by SHA-256;
the FluidAudio-managed models are the documented exception.

### Packaging

- Every product is consumable as a versioned SwiftPM dependency. This did not
  hold until shortly before the tag: `-warnings-as-errors` was applied through
  `.unsafeFlags`, and two dependencies were pinned by revision. Both bar a
  package from being depended on by version, and neither shows up in this
  package's own build - only in a consumer's resolution.
  - `-warnings-as-errors` is now opt-in through `AUDIOTOOL_STRICT_WARNINGS=1`,
    which CI sets.
  - FluidAudio moved from a revision pin to upstream 0.15.5. The pin predated
    any release carrying the mel-context chunk-boundary fix; releases from 0.10
    on carry it.
  - MisakiSwift moved from a revision pin to the fork tag `1.0.5-static.1`,
    naming the same commit. The fork exists only to drop `type: .dynamic` from
    the manifest; upstream 1.0.5 and 1.0.6 both still declare it.
  - `Scripts/check-publishable.sh` fails the build if either property returns.

### CI

- A third job runs real models: the FluidAudio suites (84 tests - VAD,
  transcription including the chunk-boundary regression, diarization, Sortformer,
  streaming) against weights it downloads and caches, plus an end-to-end CLI run
  of FRCRN over a tracked fixture. Nothing before it loaded a checkpoint, and
  three `SortformerComparisonTests` were failing unnoticed as a result.
- `Scripts/smoke-cli.sh` and `Scripts/check-audio-output.py` are that CLI check:
  they assert the output is the right rate and length, is not silence, and is not
  a copy of the input - the failures that still leave a plausible WAV behind.
- Pyannote's offline diarizer throws `noSpeechDetected` on the `mix*` fixtures -
  on the current and previous FluidAudio pins alike, so it is standing upstream
  behaviour rather than a regression. Not a bandwidth problem, despite the
  fixtures' 8 kHz origin: the same speech band-limited to 4 kHz still diarizes,
  and the 16 kHz mixture throws too. What they share is being synthetic,
  continuously overlapped mixtures. The tests that hit it never used the Pyannote
  result, so they now ask only Sortformer - and assert that it found speech,
  which `segmentCount >= 0` never did.

### Platforms

- iOS 18 actually builds. It was declared in the manifest and advertised in the
  README, but `AudioToolCore` reached for `homeDirectoryForCurrentUser`, which is
  unavailable under the iOS SDK, so every backend failed to compile. Cache roots
  now resolve through `FileManager.userHomeDirectory`, which is the app container
  on iOS - where `HubApi` already puts its downloads - and the real home on macOS.
- CI builds all nine library products for `generic/platform=iOS` alongside the
  macOS test job, so the declared deployment target is checked rather than
  assumed.

### CLI

- `--weights` is honored by every model mode. `se48k-bg`, the three
  speaker-separation modes and `sr48k` parsed it and then constructed a
  downloading provider, so a local override was accepted and ignored. `sr48k`
  takes the checkpoint and reads `config.json` beside it, the layout a download
  produces, and says so if that file is missing.
- Failures print `errorDescription` rather than the interpolated error value, so
  a missing checkpoint reads as a sentence instead of `weightsNotFound("...")`.

### API notes

- `AudioToolConfiguration` carries `modelMemoryLimit` and nothing else. It
  previously advertised `precision`, `enableSTFTCache`, `modelRepository`,
  `segmentPoolSize` and `channelCapacity`; no code read any of them, so setting
  one changed nothing. Precision is a per-provider argument, repositories are
  per-model in the catalog, and chunking is a per-call parameter.
- The USS factories in `USSProviders` default to FP32, matching
  `USSMLXProvider`. They defaulted to FP16 while the provider they construct
  defaulted to FP32; FP16 there is slower, peaks higher, and costs about 68 dB
  SI-SDR, buying only download size.
- `streamSynthesis` yields incremental buffers for Apple TTS but a single
  whole-utterance buffer for Kokoro and Chatterbox. Sentence-level chunking for
  the MLX synthesizers is open work; the API shape is stable either way.
- `KokoroSwift.LSTMKernelTest` is gone from the library. It compared the fused
  LSTM Metal kernel against plain MLX ops, printed the result and returned a
  `Bool` no caller checked. The comparison now runs as `LSTMKernelTests` in the
  MLX integration suite, where a divergence fails a build.
