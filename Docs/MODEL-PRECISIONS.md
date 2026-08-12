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
| fp32 | 211 MiB | 15.6x | 1090 MiB | reference | situational - reference quality, but fp16 is faster and smaller at 70 dB |
| fp16 | 106 MiB | 24.6x | 937 MiB | 70.1 dB | **recommended** - fastest, lowest memory, half the size |
| int8 | 86 MiB | 21.2x | 968 MiB | 59.2 dB | situational - only if 20 MiB of disk matters more than 11 dB and some speed |
| int6 | 75 MiB | 21.1x | 941 MiB | 48.1 dB | not recommended - slower than fp16, 22 dB worse, saves 31 MiB |
| int4 | 64 MiB | 21.3x | 931 MiB | 39.0 dB | not recommended - slower than fp16, 31 dB worse, saves 42 MiB |

Measured on an Apple M1 Pro (16 GB) over 30 s of audio, three timed
runs per configuration. Speed is realtime factor - higher is faster.
Expect different absolute numbers on other hardware; the ordering has
been stable across the models measured so far, but is not guaranteed.

Quality compares each precision against **this model's own fp32
output**. It is what the precision costs, not how good the model is.


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
| MossFormer2 SR | **fp32** | fp16 (returns silence-as-NaN), int6, int4 - none published |
| USS ResUNet30 | **fp32** | - |
| Chatterbox TTS | **8bit** or **4bit** | fp16, 6bit |

Two things worth knowing before generalising from any single row. Smaller is not
reliably faster: fp16 is the fastest option for MossFormer2 SE and the slowest for
Chatterbox. And integer quantization reaches only the linear layers, so on
convolution-heavy models it shrinks the download without changing speed or memory
at all - MossFormer2 SR is the clearest example.
