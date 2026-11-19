# Choosing a precision

Each model here publishes more than one checkpoint. They are not
interchangeable, and the smallest is rarely the right one: on some models a
smaller checkpoint is also *slower* and uses *more* memory, because what
dominates the footprint is the forward pass rather than the weights.

These tables say what each choice costs, measured rather than estimated.

**Quality is each precision against its own model's fp32 output** - the cost of
the precision change, not a score for the model. Roughly: above 60 dB is
inaudible on speech, 40-50 dB is audible on close listening, below 40 dB is
audible.


## MossFormer2 SE 48 kHz - speech enhancement

| precision | size | speed | memory | quality vs fp32 | |
| --- | ---: | ---: | ---: | ---: | --- |
| fp32 | 211 MiB | 21.9x | 885 MiB | reference | situational - reference quality, and only 16% slower than fp16 |
| fp16 | 106 MiB | 25.5x | 888 MiB | 70.1 dB | **recommended** - fastest, half the size, 70 dB |
| int8 | 86 MiB | 21.9x | 978 MiB | 59.2 dB | situational - 0.41x the download for 59 dB - no faster than fp32, and more memory |
| int6 | 75 MiB | 22.2x | 967 MiB | 48.1 dB | not recommended - slower than fp16, 22 dB worse, saves 31 MiB |
| int4 | 64 MiB | 22.2x | 886 MiB | 39.0 dB | not recommended - slower than fp16, 31 dB worse, saves 42 MiB |

Measured on an Apple M1 Pro (16 GB) over 30 s of audio, three timed
runs per configuration. Speed is realtime factor - higher is faster.
Expect different absolute numbers on other hardware; the ordering has
been stable across the models measured so far, but is not guaranteed.

Quality compares each precision against **this model's own fp32
output**. It is what the precision costs, not how good the model is.


## Voice Isolator 48 kHz - speech isolation, offline

| precision | size | speed | memory | quality vs fp32 | |
| --- | ---: | ---: | ---: | ---: | --- |
| fp16 | 152 MiB | 3.2x | 1094 MiB | 69.9 dB | **recommended** - faster, 120 MiB smaller, 378 MiB lighter, 70 dB |
| fp32 | 272 MiB | 2.7x | 1472 MiB | reference | situational - reference output, if you need it bit-exact |

## Voice Isolator Realtime 48 kHz - speech isolation, causal

| precision | size | speed | memory | quality vs fp32 | |
| --- | ---: | ---: | ---: | ---: | --- |
| fp16 | 139 MiB | 3.7x | 1700 MiB | 72.4 dB | **recommended** - the only sensible default here |
| fp32 | 253 MiB | 3.0x | 2614 MiB | reference | not recommended - 23% slower for 914 MiB more |

These two are the clearest "use fp16" in the package after the Core ML enhancer,
and the only models here whose **default** is fp16: it is what the MLX port
computes in upstream and what it was measured against the PyTorch reference at.
fp16 is faster *and* smaller *and* lighter in both, with no quality column worth
arguing about - 70 dB is inaudible on speech.

Two things in these tables surprise people, and both are why they are measured
rather than reasoned about:

- **The causal model is the faster of the two** - 3.7x against 3.2x at fp16 -
  even though it sees less context. It is also the one you would reach for
  *last* on a recorded file, because it trades the offline model's full 8-second
  window for the ability to run on live audio.
- **The causal model is also the heavier**, by 1.6x at fp16, which is the
  opposite of what "realtime" suggests. A streaming session carries a 600-frame
  attention window with a crossfaded second lane plus Mamba state, and that
  costs more than the offline path's bounded per-window activations. If memory
  is the constraint and the audio is already recorded, the offline model is the
  cheaper choice on both counts.

Both are flat in input length: doubling the input to 60 s moves the peak by 4%
(offline) and 1% (realtime), and leaves the rate unchanged. The offline model's
activations barely move at all - 503, 545 and 555 MiB at 8, 30 and 60 s - because
each 8-second window is an independent forward pass.

**These are not the numbers on the port's own README, and the difference is the
quantity, not the result.** VoiceIsolatorMLX reports *activations for one
8-second window*: +430 MiB at fp16 and +714 MiB at fp32, on an M4. The
comparable figure here is MLX's own activation accounting, which over the same
8-second window on this M1 Pro reads 503 MiB and 949 MiB. The column above is
something else - the whole process footprint, which is what a memory budget is
charged and what Activity Monitor shows. For offline fp16 it decomposes as
roughly 503 MiB of activations, 152 MiB of resident weights, and ~400 MiB that
MLX does not count: Metal's library and driver allocations, the Swift runtime,
and the CPU-side buffers either side of the model.

That last gap is not particular to this model, and it is not a number to read
closely: across the reports in `BenchmarkResults/` the same subtraction runs
from -251 MiB to +528 MiB, because the process footprint and MLX's high-water
counter are sampled independently.

The realtime model has no upstream memory figure to disagree with - the README
publishes its per-push latency and nothing else - so 1.5-1.7 GiB is new
information rather than a discrepancy. Its activations really are about 2.3x the
offline model's on the same window (1136 MiB against 503 at fp16), and that is
the reference's two-lane crossfade doing exactly what it is supposed to: every
600 frames the session clones its layer states and runs the whole network at
batch 2 for the next 200, holding two attention caches, then swaps lanes and
drops the old one. Double the forward pass for a quarter of the time is close to
the 2.3x measured. It is bounded - the retired lane is released at every
boundary - and it is not something this package can trim without diverging from
the reference, so it is reported rather than optimised.

Speed and memory measured on an Apple M1 Pro (16 GB) over 30 s of audio through
`audio-tool-bench`, five timed runs per configuration in an isolated process.
Quality is the SNR of each fp16 output against the same model's fp32 output over
`Tests/AudioToolFluidAudioTests/Fixtures/test_48k.wav` (6.25 s of VCTK speech),
run through `audio-tool`. As everywhere in this document it is what the
precision costs, not how good the model is.

## MossFormer2 SR - super resolution, 16 -> 48 kHz

| precision | size | speed | memory | quality vs fp32 | |
| --- | ---: | ---: | ---: | ---: | --- |
| fp32 | 418 MiB | 2.1x | 4149 MiB | reference | **recommended** - quantization changes neither speed nor memory here |
| int8 | 293 MiB | 2.1x | 3985 MiB | 62.5 dB | situational - 30% smaller for 62 dB, if download size is the constraint |
| int6 | 282 MiB | 2.1x | 3975 MiB | 50.8 dB | not published - no speed or memory gain, 51 dB |
| int4 | 272 MiB | 2.0x | 4058 MiB | 38.1 dB | not published - no speed or memory gain, 38 dB |

**Only fp32 and int8 are published.** The remaining rows are measurements
rather than downloads - int6 (strictly dominated by int8); int4 (strictly
dominated by int8). They are kept because the comparison is the point: it is
what shows what quantizing this model does and does not buy.

Measured on an Apple M1 Pro (16 GB) over 30 s of audio, three timed
runs per configuration. Speed is realtime factor - higher is faster.
Expect different absolute numbers on other hardware; the ordering has
been stable across the models measured so far, but is not guaranteed.

Quality compares each precision against **this model's own fp32
output**. It is what the precision costs, not how good the model is.


## USS ResUNet30 - universal source separation

| precision | size | speed | memory | quality vs fp32 | |
| --- | ---: | ---: | ---: | ---: | --- |
| fp32 | 102 MiB | 27.2x | 1203 MiB | reference | **recommended** - faster and lower memory than fp16 |
| fp16 | 51 MiB | 25.8x | 1259 MiB | 57.3 dB | situational - half the download, but slower, more memory and 57 dB |

Measured on an Apple M1 Pro (16 GB) over 30 s of audio, three timed
runs per configuration. Speed is realtime factor - higher is faster.
Expect different absolute numbers on other hardware; the ordering has
been stable across the models measured so far, but is not guaranteed.

Quality compares each precision against **this model's own fp32
output**. It is what the precision costs, not how good the model is.

## Chatterbox TTS - multilingual speech synthesis

| precision | size | speed | memory | |
| --- | ---: | ---: | ---: | --- |
| fp32 | 2580 MiB | 1.06x | 4017 MiB | situational - best quality, heaviest |
| fp16 | 1293 MiB | 0.67x | 3058 MiB | **not recommended** - 37% slower than fp32 |
| 8bit |  917 MiB | 0.99x | 2287 MiB | **recommended** - fp32 speed, 1.7 GiB less memory |
| 6bit |  769 MiB | 0.86x | 2250 MiB | not recommended - slower than both 8bit and 4bit |
| 4bit |  620 MiB | 1.05x | 2205 MiB | **recommended** - fastest and smallest |

Speed here is seconds of audio generated per second of wall time, so it is not
comparable to the models above, which consume audio.

No quality column: synthesis samples, so two precisions do not produce comparable
waveforms even from the same text and seed. Voice cloning is unaffected by the
integer widths - the speaker encoder is stored at full precision in every one of
them.

## MossFormerGAN SE 16 kHz - speech enhancement, CoreML

| precision | size | speed | memory | quality vs fp32 | |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 13 MB | 3.7x | 1656 MiB | reference | not recommended - slower and far heavier |
| FP16 | 7.6 MB | 4.2x | 302 MiB | 61 dB | **recommended** - faster, 5.5x less memory |

The clearest choice of any model here. CoreML fixes precision when the package is
compiled, so these are two files rather than one model with a switch.

## Summary

| model | use | avoid |
| --- | --- | --- |
| MossFormerGAN SE (CoreML) | **FP16** | FP32 |
| MossFormer2 SE 48 kHz | **fp16** | int6, int4 |
| Voice Isolator 48 kHz | **fp16** | fp32, unless you need reference output |
| Voice Isolator Realtime 48 kHz | **fp16** | fp32 |
| MossFormer2 SR | **fp32** | fp16 (returns silence-as-NaN), int6, int4 - none published |
| USS ResUNet30 | **fp32** | - |
| Chatterbox TTS | **8bit** or **4bit** | fp16, 6bit |

Two things worth knowing before generalising from any single row. Smaller is not
reliably faster: fp16 is the fastest option for MossFormer2 SE and the slowest for
Chatterbox. And integer quantization reaches only the linear layers, so on
convolution-heavy models it shrinks the download without changing speed or memory
at all - MossFormer2 SR is the clearest example.
