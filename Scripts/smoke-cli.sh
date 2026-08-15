#!/bin/bash
#
# End-to-end smoke test: a real model, through the shipped binary, on real audio.
#
# Everything else in CI stops short of this. The unit job runs mocks, the iOS job
# only compiles, and every MLX suite that loads weights reaches for the sibling
# research checkout and skips on a runner. So nothing proved that a downloaded
# checkpoint plus a Metal device plus the chunking path actually produces audio -
# the thing every user does first.
#
# FRCRN because it is the cheapest real model: 56 MB, 16 kHz, seconds of work.
#
# Usage: Scripts/smoke-cli.sh [path-to-audio-tool]
#
# With no argument it builds and uses a debug binary, which is also what CI does -
# the point is that inference runs and produces sane audio, not how fast.
#
set -euo pipefail

cd "$(dirname "$0")/.."

binary="${1:-}"
if [ -z "$binary" ]; then
    swift build --product audio-tool
    binary=".build/debug/audio-tool"
fi

if [ ! -x "$binary" ]; then
    echo "error: no audio-tool binary at $binary"
    exit 1
fi

input="Tests/AudioToolFluidAudioTests/Fixtures/mix_16k.wav"
output="$(mktemp -d)/enhanced.wav"

echo "Running FRCRN over $input"
"$binary" --model frcrn --input "$input" --output "$output"

# A written file is not a passed test: an all-zero WAV of the right length would
# satisfy `test -f`, and that is exactly what a silently failing model produces.
PYTHONDONTWRITEBYTECODE=1 python3 Scripts/check-audio-output.py "$input" "$output"
