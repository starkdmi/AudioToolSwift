#!/usr/bin/env python3
"""Assert that an enhancement run actually produced enhanced audio.

Used by Scripts/smoke-cli.sh. Kept separate so the checks are readable and can be
pointed at any input/output pair:

    python3 Scripts/check-audio-output.py noisy.wav enhanced.wav

Five failures it is meant to catch, all of which leave a file behind and would
pass a `test -f`:

  * the wrong sample rate or channel count, which means the loader or saver was
    misconfigured
  * a truncated or duplicated signal, which is what a broken chunk reassembly
    produces
  * NaN or infinite samples, which is what an overflowing forward pass produces
    and which every threshold below would otherwise wave through
  * silence, which is what an unloaded or NaN-poisoned model produces
  * an unchanged copy of the input, which is what a model whose weights never
    reached the graph produces
"""

import array
import math
import struct
import sys


def read(path):
    """Minimal RIFF reader.

    The CLI writes 32-bit float WAV and the fixtures are 16-bit PCM; the standard
    library's `wave` refuses the former outright ("unknown format: 3").

    Returns (sample rate, channels, samples normalised to -1...1).
    """
    with open(path, "rb") as handle:
        data = handle.read()

    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"error: {path} is not a WAV file")

    offset = 12
    fmt = None
    payload = None
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        body = data[offset + 8:offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body)
        elif chunk_id == b"data":
            payload = body
        offset += 8 + size + (size % 2)  # chunks are word-aligned

    if fmt is None or payload is None:
        sys.exit(f"error: {path} has no fmt or data chunk")

    tag, channels, rate, _, _, bits = fmt
    if tag == 1 and bits == 16:
        values = array.array("h")
        values.frombytes(payload[:len(payload) - len(payload) % 2])
        scale = 32768.0
    elif tag == 3 and bits == 32:
        values = array.array("f")
        values.frombytes(payload[:len(payload) - len(payload) % 4])
        scale = 1.0
    else:
        sys.exit(f"error: {path} is format {tag} at {bits} bits, which this check cannot read")

    return rate, channels, [value / scale for value in values]


def main(source, result):
    in_rate, in_channels, in_samples = read(source)
    out_rate, out_channels, out_samples = read(result)

    problems = []

    if out_rate != in_rate:
        problems.append(f"sample rate {out_rate} against input {in_rate}")

    if out_channels != in_channels:
        problems.append(f"{out_channels} channels against input {in_channels}")

    in_frames = len(in_samples) // in_channels
    out_frames = len(out_samples) // out_channels
    if abs(out_frames - in_frames) > in_rate // 10:
        problems.append(f"length {out_frames} frames against input {in_frames}")

    # Before anything numeric. NaN loses every comparison it takes part in, so a
    # NaN-poisoned buffer slips past the silence check (`nan < 0.01` is false) and
    # the similarity check alike, and gets reported as "peak nan". An overflowing
    # fp16 forward pass is a real way to produce exactly that.
    nonfinite = sum(1 for sample in out_samples if not math.isfinite(sample))
    if nonfinite:
        problems.append(
            f"{nonfinite} of {len(out_samples)} samples are NaN or infinite"
        )
        for problem in problems:
            print(f"error: {problem}")
        return 1

    peak = max((abs(sample) for sample in out_samples), default=0.0)
    if peak < 0.01:
        problems.append(f"output is silent or near-silent (peak {peak:.5f})")

    compared = min(len(in_samples), len(out_samples))
    unchanged = sum(
        1 for index in range(compared)
        if abs(in_samples[index] - out_samples[index]) < 1e-4
    )
    if compared and unchanged / compared > 0.99:
        problems.append("output matches the input - the model did nothing")

    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 1

    print(
        f"OK: {out_frames / out_rate:.2f}s at {out_rate} Hz, peak {peak:.3f}, "
        f"{unchanged / compared:.1%} of samples unchanged"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: check-audio-output.py <input.wav> <output.wav>")
    sys.exit(main(sys.argv[1], sys.argv[2]))
