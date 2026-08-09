# Parity

Evidence that the Swift ports compute what the reference implementations
compute. Not a benchmark, and not a smoke test - a fixed input, a recorded
reference output, and a number.

## The split

**Generation** needs Python, MLX, the reference code under `Models/python/` and
the weights. It runs on the one machine that has all of that, rarely, on
purpose. Nothing in this directory is a dependency of the Swift package.

**Verification** needs Swift, the weights and the artifacts. It runs anywhere.
`swift test` stays hermetic; the parity suite is opt-in and skips when the
artifacts are absent, the same shape as `TestGate`.

## Running generation

The `mlx` conda environment, not `edge-ml` - the latter's torchaudio will not
load, and the reference runners read audio through it.

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/mlx/bin/python Parity/generate.py --list
/opt/homebrew/Caskroom/miniforge/base/envs/mlx/bin/python Parity/generate.py --all
```

Output lands in `Parity/artifacts/` as `<case>.safetensors` plus a `<case>.json`
sidecar. safetensors because Swift already reads it - verification needs no new
parser and no new dependency.

## What an artifact holds

The **input tensor**, not a path to a wav. Decoding the same file through
soundfile and through AVFoundation can differ in the last bit or two, and a
harness that reports that as model divergence is worse than none. Swift reads
float32 samples straight out of the artifact.

Then one named tensor per output, cut at the seams that matter:

| Case | Seams |
|---|---|
| `mossformer2_sr_48k` | `upsampled_48k` (librosa's 16→48 kHz), `model_output`, `enhanced` (after `bandwidth_sub`) |
| `mossformer_gan_se_16k_coreml` | `enhanced_segment` (model + STFT), `enhanced_full` (adds stitching) |
| `mossformer2_ss_*` | `speaker_N` raw, `speaker_N_normalized` as the API returns it |
| `demucs_vocals_44k` | every stem, per channel |
| `uss_resunet30_32k` | one output per query condition |
| `chatterbox_conditionals_*` | `ve_speaker_emb`, both token tensors, `s3gen_prompt_feat`, `s3gen_embedding` |

Chatterbox is cut differently from the rest and worth a note. Everything past
`prepare_conditionals` samples, so the comparison stops there rather than at an
output: the seams are the conditioning tensors, and the two token ones are compared
for exact equality because they are codebook indices, where SNR is only the harness
applying one metric uniformly. Two cases, because one clip cannot exercise it — a
3 s prompt already at 24 kHz leaves both conditioning windows and the resampler
idle, so `chatterbox_conditionals_22k_long` runs a 12 s prompt at 22050 Hz through
the truncations and a 160/147 polyphase ratio.

It is also the one case where generation found a defect in the *reference* rather
than the port; `Tests/AudioToolParityTests/ParityThresholds.swift` has the account.

A single output-level number gives pass/fail. These give *where*, without
needing internal taps in production code.

The sidecar pins the weights hash, source-audio hash, the reference module's own
hash, the package revision and every library version. Paths are checkout-relative;
these get published, and an absolute `/Users/...` in a public file is noise.

## The metric

SNR in dB of the Swift output against the reference, which is what the
conversion work already used: `Models/python/frcrn_se_mlx/README.md` records
115.98 dB for MLX against PyTorch. float32 round-off alone lands around 110-130
dB. Anything under ~60 dB is a real difference.

Thresholds are derived from a first run and recorded, not invented.

**The 23.8 dB in the CoreML GAN README is not a tolerance.** That is CoreML
against PyTorch - conversion loss, measured and accepted when the 256-frame
variant was chosen. Both sides of the GAN case run the *same* `.mlpackage`
through the *same* MLX STFT, so its target is round-off like the rest. A 24 dB
result there means the Swift wrapper's framing or stitching is wrong.

## Reference

MLX Python, under `Models/python/` - the implementation the Swift was ported
from. The PyTorch originals exist too (`~/Code/Languages/Python/AI/**/
last_best_checkpoint*.pt`), and PyTorch-vs-MLX is a separate, looser number that
was already paid for at conversion time.

Adapters import the reference modules in place. Nothing third-party is vendored
into this repository.

## Guards

A case whose input is near-silent agrees with almost any implementation while
looking green. `ParityCase` refuses to write one - `MIN_INPUT_RMS` in
`harness.py`. This caught a real mistake: USS's `test_speech.wav` opens on six
seconds of silence, and the first draft of that case had a 2e-5 RMS input and
two "separated" outputs that agreed with each other and with nothing else.
