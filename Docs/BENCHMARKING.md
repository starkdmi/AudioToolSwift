# Benchmarking

`audio-tool-bench` measures what each model in this package costs on the machine
it is run on: how long it takes to load, how long an inference takes, and how much
memory it holds while doing it. It writes a JSON report and a markdown summary,
both carrying the machine and the settings, so runs from different machines can be
compared without a conversation about what was on screen at the time.

## Build and run

Same constraint as `audio-tool`: MLX needs a compiled Metal shader library, so
build with `xcodebuild` and run the binary **from the products directory**. See
AGENTS.md, "Metal/MLX", for why.

```bash
xcodebuild build -scheme audio-tool-bench -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet

.build/DerivedData/Build/Products/Release/audio-tool-bench --list
```

`-configuration Release` is not optional. A debug build of this code measures the
optimiser; the report says so in a note, but the numbers are still wasted.

The SwiftPM path works too, once the metallib is staged:

```bash
./Scripts/build_mlx_metallib.sh release
swift build -c release --product audio-tool-bench
./.build/release/audio-tool-bench --list
```

## First run on a new machine

Weights download from HuggingFace on first use, and a load time with a download in
it is not a load time. Fetch everything once, then measure:

```bash
audio-tool-bench --prefetch
audio-tool-bench -t "$(git rev-parse --short HEAD)"
```

The report flags any case whose weights arrived during the run, so forgetting this
produces a warning rather than a quietly wrong number.

"Cached" here means *the files the model actually opens* are on disk. That is
narrower than the download manifest, and the distinction is `ModelFiles.standard`
(what to fetch) against `ModelFiles.standardRequired` (what must be present to
load). Only the super-resolution provider reads a `config.json`;
`MossFormer2SE48KProvider` builds its `PipelineConfiguration` in code,
`MossFormer2SSProvider` hardcodes its `MossFormer2Config`, and FRCRN calls nothing
but `loadWeights`.

Writing this benchmark is what surfaced that: the providers used to gate their
*presence* check on the download manifest too, so a machine holding the safetensors
without the unused config was treated as having nothing and paid a repository
round-trip on every load - and would have failed outright with no network. Both
sides ask the narrower question now, and `ModelCatalogAgreementTests` pins them
together.

If a weights file is on this machine already, the benchmark uses it directly rather
than fetching its own copy - see `--weights-root` below. `weightsSource` records
which of the three routes each case took (`local`, `cache`, `network`), because a
load time from a local file, a warm cache and a cold download are three different
measurements and only two of them belong in a comparison.

## What it runs

Twenty-two cases, MLX and CoreML. Where a model publishes more than one
precision, each is its own case — that comparison is what `MODEL-PRECISIONS.md`
is built from:

| id | model |
| --- | --- |
| `mlx.frcrn_se_16k` | FRCRN speech enhancement, 16 kHz |
| `mlx.mossformer2_se_48k.` `fp32` / `fp16` / `int8` / `int6` / `int4` | MossFormer2 speech enhancement, 48 kHz |
| `coreml.mossformer_gan_se_16k.fp32` / `.fp16` | MossFormerGAN enhancement, CoreML |
| `mlx.mossformer2_ss.2spk` | Speaker separation, 2 speakers, 16 kHz |
| `mlx.mossformer2_ss.3spk` | Speaker separation, 3 speakers, 8 kHz |
| `mlx.mossformer2_ss.2spk_whamr` | Speaker separation, WHAMR, 8 kHz |
| `mlx.mossformer2_sr_48k.fp32` / `.int8` | Super resolution, 16 kHz in, 48 kHz out |
| `mlx.demucs.vocals` | Demucs, vocals stem only |
| `mlx.demucs.all_stems` | Demucs, all four stems |
| `mlx.uss.fp16` / `.fp32` | USS ResUNet30 |
| `mlx.chatterbox.` `fp32` / `fp16` / `8bit` / `6bit` / `4bit` | Chatterbox TTS |

Each model lists the precisions it publishes, so this table changes when a
checkpoint is uploaded or withdrawn. Super resolution is the one to read twice:
fp16 is absent because its forward pass is all-NaN, and int6/int4 because both are
strictly dominated by int8 and were never uploaded.

Chatterbox is measured in a column of its own: its RTF counts audio *generated*
per second of wall time, where every other case counts audio *consumed*. The two
denominators are not comparable, which is why `rateBasis` exists rather than the
number simply being dropped in beside the others.

The FluidAudio-backed VAD, transcription and diarization providers are deliberately
absent: they wrap a third-party pipeline, and their numbers describe that project
rather than this one. That one is a small addition to `BenchmarkCatalog` if the
question ever calls for it.

Both CoreML cases download like every other one. CoreML fixes precision when the
package is compiled, so FP32 and FP16 are two separate files rather than one model
with a switch — which is why they are two cases and not one.

To measure a `.mlpackage` you built yourself instead of the published one:

```bash
audio-tool-bench --coreml-gan /path/to/MossFormerGAN_256frames.mlpackage
# or
export AUDIOTOOL_BENCH_COREML_GAN=/path/to/MossFormerGAN_256frames.mlpackage
```

That overrides the FP32 case only, since it names one file. Until 2026-08-12 it was
required rather than optional — the packages had no repository, so without a path
the whole case reported as *skipped*, and FP16 was measured only when its package
happened to sit beside the FP32 one in a checkout.

## Weights already on the machine

`--weights-root <dir>` points at a checkout holding weights, and defaults to the
sibling research checkout when one is present - so on a development machine eleven
of the twelve cases run with no downloads at all. Absent in a standalone clone, in
which case everything resolves through HuggingFace exactly as before.

Existence is not enough to be used: a file under a megabyte is treated as a stub
rather than weights. That is not hypothetical - the checkout's
`resunet30_fp16.safetensors` is a 1152-byte macOS bookmark alias, and taking it at
face value failed inside MLX with "Invalid json header length", a message about
entirely the wrong layer. Symlinks are resolved before the size is read, since the
checkout stores `resunet30_fp32.safetensors` as one.

## One model at a time

Every case runs in its own child process, sequentially, with a cooldown between
them. This is the central design choice and it is what makes the numbers usable on
a 16 GB machine:

- **Peak memory means something.** MLX's allocator is process-global and its cache
  survives `unload()`. Two models measured in one process share a high-water mark,
  and the second inherits the first's.
- **A bad case is contained.** A model that exhausts memory, traps, or wedges Metal
  takes down one case. The run continues and the report says which one died.
- **Cold start is real.** Graph compilation, Metal pipeline construction and the
  weights cache are per-process. Measuring the second model's "first inference" in
  an already-warm process measures something no user ever pays.

`--in-process` turns this off. It is faster and the memory columns stop meaning
anything; the report carries a note saying so.

The cooldown (`--cooldown`, default 5 s) exists because a laptop that runs twelve
models back to back is measuring the twelfth under the thermal load of the first
eleven. Raise it for a long run:

```bash
audio-tool-bench --cooldown 30
```

The report records the thermal state at the start and end of every case and flags
any case where it moved.

## Memory limits

MLX's defaults scale with the machine rather than the work. The memory limit
defaults to 1.5x the GPU's recommended working set, and the **cache limit defaults
to the memory limit** — so on a 16 GB Mac a long chunked job's resident size climbs
until the OS pushes back, and the peak recorded for a model describes the host.

Two caps are applied before any model touches MLX:

| cap | default | flag |
| --- | --- | --- |
| MLX allocator cache | 512 MiB — what this package applies in production | `--gpu-cache-limit <MB>` |
| MLX total allocation | 60% of RAM, capped by the GPU's recommended working set | `--gpu-memory-limit <MB>` |

Both defaults match production deliberately: `MLXCachePolicy` and
`USSInference` install the same two limits, so the benchmark measures shipped
behaviour rather than a configuration nobody runs.

**The library wins on any chunked path.** `MLXCachePolicy.applyProcessLimits()`
runs inside every provider's `load()`, so whatever `--gpu-cache-limit` asked for is
replaced before the first chunk. That is deliberate — it is what keeps a host app's
peak bounded — but it means the requested value and the value in force are
different questions. Both are recorded: `gpuCacheLimitBytes` is what was asked for,
`effectiveGpuCacheLimitBytes` is read back from MLX after the case. When they
disagree, believe the second.

`mlx.uss.*` names the cache cap explicitly, because `USSInference` applies its own
at construction regardless. Recording it keeps the report honest about what was in
force rather than implying the run-wide default was.

Before a case starts, its declared working set is checked against the machine. A
case that cannot fit is skipped with a reason; one that would take more than half
the machine prints a warning and runs.

## Input

Synthetic by default: a harmonic stack under a syllable-rate envelope plus a little
hiss, generated from a fixed seed. Every machine measures the same samples without
downloading anything, which is the point — the committed fixtures are licensed test
inputs, not benchmark material, and a benchmark that needs a file only one checkout
has cannot travel.

This is sound because inference cost here is a function of length and rate, not
content: fixed-size chunks, no early exit, no data-dependent branching. What
synthetic input cannot measure is output *quality* — that is
`Tests/AudioToolParityTests`, which compares against recorded reference tensors.

For a specific recording:

```bash
audio-tool-bench --input ~/audio/noisy.wav --seconds 60
```

The file is resampled per case to that model's rate, and looped rather than
zero-padded if it is shorter than `--seconds`.

**`--seconds` changes the answer.** Most providers process short input in one pass
and chunk anything longer, and chunking with overlap costs more model invocations
per second of audio. The default of 30 s is past every chunking threshold in the
catalog, so it measures the path a real file takes. A 4-second run measures the
direct path and is not comparable to it.

## Reading the report

Both files land in `--out` (default `./BenchmarkResults`, gitignored), named after
the machine, the timestamp and the first eight characters of the run id — the
timestamp is only second-accurate, so without the id two short runs would collide
and the second would overwrite the first in silence.

**Speed.** `load` is `load()`. `first` is the first inference, which carries MLX
graph compilation and is routinely several times the steady state — it is reported
separately rather than folded into the RTF. `median` is the median of the timed
iterations, and RTF is audio seconds per wall second computed from it. Median
rather than minimum: unlike a pure numeric kernel these runs include cache trimming
and allocator behaviour that a real caller also pays. The best-of-N RTF is in the
next column for anyone who wants it.

**Memory.** `peak` is the process `phys_footprint` — what Activity Monitor calls
Memory, and what the OS charges against a memory limit. `weights` is the footprint
growth across `load()`.

The `mlx` columns are MLX's own accounting. `mlx peak` is total MLX memory at its
highest during the timed runs, **weights included**: the peak counter is reset
after load, but that zeroes the high-water mark without freeing anything, so the
next allocation restores it to at least the weights. What the reset does buy is
dropping the load-time spike — conversion, unflattening, `verify: .all` — which is
real but is not what an inference costs. `mlx act.` subtracts the resident weights
and is an *estimate* of the activations, not a measurement.

`mlx act.` shows `-` and `mlx weights` shows `lazy` when the provider had not
materialised its weights by the time `load()` returned. MLX arrays are lazy, and
not every provider evals at load:

| provider | at load | effect |
| --- | --- | --- |
| `FRCRNSE16KProvider` | dummy forward pass + `eval` | weights land in `load` |
| `MossFormer2SSProvider` | `eval(candidate)` | weights land in `load` |
| `MossFormer2SR48KProvider` | evals | weights land in `load` |
| `MossFormer2SE48KProvider` | `loadWeights` only, no `eval` | weights land in `first` |

For a lazy provider, `load` excludes weight materialisation and `first` includes
it, so those two columns are not the same split of work as the other rows'. Total
time and RTF are unaffected. The report says so in a note rather than leaving the
columns to be compared as if they matched.

**CPU util** below 1 on a GPU model means the CPU was waiting on Metal. That is the
expected shape, and it is why a wall-clock RTF alone cannot tell a GPU win from a
CPU one.

**Notes** collects everything that should make you distrust a row: a debug build, a
cold download folded into a load time, a thermal state that moved mid-case, a
machine on battery, Low Power Mode, and every skip and failure with its reason.

## Comparing runs

The JSON is the record; `schemaVersion` is bumped whenever a field changes meaning,
and case ids are an interface — renaming one breaks comparison with older reports.
Every report carries the resolved revisions of mlx-swift, SwiftAudio and the other
dependencies that decide the numbers, read from `Package.resolved` at runtime.
A binary copied away from its checkout reports that field as absent rather than
guessing.

Tag a run with whatever you will want to know later:

```bash
audio-tool-bench -t "before-chunk-rewrite"
audio-tool-bench -t "after-chunk-rewrite"
```

`--redact-host` prepares a report for publication. It removes the host name,
reduces a `--input` path to its basename, and rewrites the home directory as `~`
wherever it appears — including in skip and failure messages, which is where a
`--coreml-gan` path would otherwise show up. Best-effort, not a guarantee: a path
deliberately outside the home directory survives, so read a redacted report before
publishing it.

## Useful invocations

```bash
# What can this machine run?
audio-tool-bench --list

# Just the enhancers, longer input, more samples
audio-tool-bench -f enhancement -s 60 -n 10

# One heavy case with a long cooldown
audio-tool-bench -c mlx.demucs.all_stems --cooldown 30

# Everything, tagged
audio-tool-bench -t "$(git rev-parse --short HEAD)"
```

`--filter` matches on id, category or backend, so `-f mlx`, `-f separation` and
`-f demucs` all do what they look like.

## Relationship to the test-suite benchmarks

`ResamplerBenchmarkTests` and `USSPrewarmBenchmarkTests` stay where they are. They
measure things below a model — a resampling kernel, an embedding-swap strategy —
which need `@testable` access and a fixture, and are gated on `RUN_BENCHMARKS=1`.
`audio-tool-bench` measures whole models through their public API, which is a
different question and does not belong in a test process.
