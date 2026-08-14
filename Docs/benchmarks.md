# Benchmarks

What each model costs on one machine. These are the numbers behind the README's
table; [running-benchmarks.md](running-benchmarks.md) explains how to produce your own, and
[model-precisions.md](model-precisions.md) compares precisions of the same model.

**Machine:** Apple M1 Pro, 8 cores (6P/2E), 16 GB, macOS 26.5.
**Method:** 30 s of synthetic audio at each model's own rate, one warm-up run and
three timed runs, median reported. Every case runs in its own child process with
an 8 s cooldown, under the same MLX limits the package applies in production
(512 MiB allocator cache, memory ceiling at 60% of RAM). Weights are prefetched,
never downloaded during a run.

**RTF** is audio seconds per wall-clock second: 21.9x means a minute of audio
takes 2.7 s. **Peak** is the process footprint — what Activity Monitor calls
Memory, and what the OS charges against a memory limit.

## Transformation models

| Case | Rate | RTF | Median | Peak | Measured |
| --- | ---: | ---: | ---: | ---: | --- |
| `mlx.uss.fp32` | 32 kHz | 28.8x | 1.042 s | 1202 MiB | 2026-08-10 |
| `mlx.mossformer2_se_48k.fp16` | 48 kHz | 25.5x | 1.176 s | 887 MiB | 2026-08-13 |
| `mlx.mossformer2_se_48k.fp32` | 48 kHz | 21.9x | 1.371 s | 884 MiB | 2026-08-13 |
| `mlx.demucs.vocals` | 44.1 kHz | 13.4x | 2.232 s | 3234 MiB | 2026-08-10 |
| `mlx.frcrn_se_16k` | 16 kHz | 8.7x | 3.452 s | 2503 MiB | 2026-08-12 |
| `coreml.mossformer_gan_se_16k.fp16` | 16 kHz | 4.2x | — | 302 MiB | see below |
| `coreml.mossformer_gan_se_16k.fp32` | 16 kHz | 3.7x | — | 1656 MiB | see below |
| `mlx.demucs.all_stems` | 44.1 kHz | 3.4x | 8.942 s | 3778 MiB | 2026-08-10 |
| `mlx.mossformer2_ss.2spk_whamr` | 8 kHz | 3.2x | 9.271 s | 3833 MiB | 2026-08-12 |
| `mlx.mossformer2_ss.3spk` | 8 kHz | 3.1x | 9.675 s | 4129 MiB | 2026-08-10 |
| `mlx.mossformer2_sr_48k.fp32` | 16 → 48 kHz | 1.9x | 15.490 s | 4144 MiB | 2026-08-12 |
| `mlx.mossformer2_sr_48k.int8` | 16 → 48 kHz | 1.9x | 15.497 s | 3983 MiB | 2026-08-12 |
| `mlx.mossformer2_ss.2spk` | 16 kHz | 1.7x | 18.040 s | 5428 MiB | 2026-08-12 |

Three things are worth reading off this table before any single number:

- **Speaker separation and super-resolution have almost no headroom.** At
  1.7–1.9x they still beat realtime, but 30 s of audio takes 15–18 s and they
  hold 4–5 GiB doing it — an order of magnitude tighter than the enhancers above
  them, and enough to make them batch operations on this hardware rather than
  live ones.
- **Memory does not follow the checkpoint.** FRCRN is a 53 MiB download that
  peaks at 2.5 GiB; MossFormer2 SE is four times the download at a third of the
  footprint. What dominates is the forward pass.
- **The Core ML enhancer's FP16 package is the one unambiguous precision win**
  in the package: faster than FP32 and 5.5x lower peak, because the FP32 path
  allocates ~1.4 GiB of transients that get compressed. Resident memory is
  identical between the two.

The two Core ML rows come from the measurement published on the model card, taken
on this machine after the Float16 output fix; their report file was not retained,
which is why they carry no median here.

`BenchmarkResults/` is gitignored — these reports describe one machine on one
day, and tracking every one of them would age badly. The exception is the report
behind the MossFormer2 SE ladder, which is committed because it is the evidence
for a correction:
`BenchmarkResults/bench-MacBookPro18-3-20260813-173110-9669e0df.{json,md}`,
host-redacted. Older rows were measured under the same method but their reports
are local only; run the harness yourself and you get your own.

## Precisions

MossFormer2 SE, MossFormer2 SR, USS and Chatterbox each publish several
checkpoints, and the smallest is rarely the right one — on some models a smaller
checkpoint is slower and uses more memory. [model-precisions.md](model-precisions.md)
has a table per model with size, speed, memory and quality against that model's
own fp32 output.

The short version: use **fp16** for MossFormer2 SE, **fp32** for MossFormer2 SR
and USS, **FP16** for the Core ML enhancer, and **8bit** or **4bit** for
Chatterbox.

## Text to speech

TTS is measured separately because its rate has a different denominator: seconds
of audio *generated* per second of wall time, where every case above counts audio
*consumed*. Chatterbox at fp32 runs at 1.06x and at 8bit at 0.99x with 1.7 GiB
less peak memory. The per-precision table is in
[model-precisions.md](model-precisions.md).

## What is not measured here

Transcription, voice activity detection and diarization run through
[FluidAudio](https://github.com/FluidInference/FluidAudio), and Apple's Speech,
Translation and speech-synthesis providers run through system frameworks.
Benchmarking either would describe those projects rather than this one. Adding
them is a small change to `BenchmarkCatalog` if the question comes up.

## Reading these numbers on other hardware

Absolute RTF is a property of the machine. The durable claims are the orderings
and the ratios — that super-resolution is an order of magnitude more expensive
than enhancement, that fp16 beats fp32 on MossFormer2 SE, that quantizing
MossFormer2 SR buys download size and nothing else.

Two practical notes for anyone re-measuring, both learned the hard way:

- **Prefetch, then measure.** Inference taken minutes after a large download
  competes with page-cache and indexing work and can read 27% slow. The harness
  flags a case whose weights arrived during the run; treat that row as void.
- **Read the standard deviation before trusting a median.** These cases sit
  between 0.002 s and 0.12 s. Anything above ~0.5 s is contaminated, and the
  harness prints it.

Ordinary desktop load is *not* a problem: four unchanged cases re-measured two
days apart, with applications open, reproduced their medians to within 3%.

## History

Rows are carried forward from the run that measured them rather than re-run for
each release, and the date column says which. A row is re-measured when code
underneath it changes — the super-resolution rows were re-taken after four
SwiftAudio changes to chunk assembly and filter design (which turned out to cost
nothing), and the MossFormer2 SE ladder was re-taken in one clean session on
2026-08-13 to replace a download-contaminated fp32 reading of 15.6x.
