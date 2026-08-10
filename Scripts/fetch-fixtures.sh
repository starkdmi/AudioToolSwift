#!/usr/bin/env bash
#
# Fetch and derive the redistributable test audio fixtures.
#
# The multi-speaker fixtures use three individual English Voice-Zero samples
# mirrored by Kyutai. Kyutai identifies that directory as CC0, and the source
# revision and file hashes below are pinned so a changed upstream file cannot be
# incorporated silently. Only the three small WAV files are downloaded; no
# dataset archive is required.
#
# `test.wav`, `test_48k.wav`, and `sr_input_16k.wav` are not regenerated here.
# They remain the VCTK inputs used during model conversion validation and are
# covered by the attribution in Docs/licenses.md.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FA="Tests/AudioToolFluidAudioTests/Fixtures"
US="Tests/AudioToolUSSTests/Fixtures"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$FA" "$US"

command -v curl >/dev/null || { echo "error: curl required" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "error: ffmpeg required (brew install ffmpeg)" >&2; exit 1; }
command -v shasum >/dev/null || { echo "error: shasum required" >&2; exit 1; }

KYUTAI_REV="323332d33f997de8394f24a193e1a76df720e01a"
KYUTAI_BASE="https://huggingface.co/kyutai/tts-voices/resolve/$KYUTAI_REV/voice-zero"

download_checked() {
    local url="$1"
    local destination="$2"
    local expected_sha256="$3"
    local actual_sha256

    curl -sfL "$url" -o "$destination"
    actual_sha256="$(shasum -a 256 "$destination" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "error: SHA-256 mismatch for $url" >&2
        echo "expected: $expected_sha256" >&2
        echo "actual:   $actual_sha256" >&2
        exit 1
    fi
}

echo "Fetching three pinned CC0 Voice-Zero samples..."
download_checked \
    "$KYUTAI_BASE/caro_davy.wav" \
    "$TMP/caro_davy.wav" \
    "40c692c005a0268a7a5b6ebae348077d3dca6a86eb6b12bd36e343bbcd71b5f6"
download_checked \
    "$KYUTAI_BASE/bill_boerst.wav" \
    "$TMP/bill_boerst.wav" \
    "be4815e4fb760ba1b78117545a260cce4a4c124c7657bc5c6127a0fef8ba661f"
download_checked \
    "$KYUTAI_BASE/peter_yearsley.wav" \
    "$TMP/peter_yearsley.wav" \
    "fbb3920fda7ae26a5a8b317ffcae1d55c0bd5d89d075205f5a52b1e924b83f51"

echo "Fetching CC0 music from Internet Archive..."
download_checked \
    "https://archive.org/download/MarchForHonor/Distant_Wonders.mp3" \
    "$TMP/music.mp3" \
    "cc56ebba3109a6c18c18c7375c88fa2043769ddacc52360a22602fc9d5fb4675"

ff() {
    ffmpeg -hide_banner -loglevel error -y "$@"
}

echo "Deriving speech fixtures..."

# Every amix below runs with `normalize=0` and an explicit 1/N gain rather than
# `normalize=1`. amix's own normalisation divides by the number of *currently
# active* inputs, and which inputs are active in the frames around end-of-stream
# depends on demuxer scheduling: the two `normalize=1` mixtures used to produce
# either of two files at random, differing over the last quarter second by up to
# 171/32768. A fixture whose hash is a coin flip is not a fixture. The 1/N gain
# reproduces the steady state where all inputs are live, so only the tails
# change, and they no longer pump.

# Sequential two-speaker dialogue with a clear turn pause for VAD coverage.
ff -i "$TMP/caro_davy.wav" -i "$TMP/bill_boerst.wav" \
    -filter_complex \
    '[0:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=1dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=500[a];[1:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=2dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=8750[b];[a][b]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.8:level=false,apad=pad_dur=0.75,atrim=0:17.5[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 16000 -ac 1 -c:a pcm_s16le "$FA/speech_dialogue.wav"
cp "$FA/speech_dialogue.wav" "$US/speech_dialogue.wav"

# Fully overlapping two-speaker mixture for 8 kHz separation models.
ff -i "$TMP/caro_davy.wav" -i "$TMP/bill_boerst.wav" \
    -filter_complex \
    '[0:a]aresample=8000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=1dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05[a];[1:a]aresample=8000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=2dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=250[b];[a][b]amix=inputs=2:duration=longest:normalize=0,volume=1/2,volume=2dB,alimiter=limit=0.8:level=false[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 8000 -ac 1 -c:a pcm_s16le "$FA/mix_8k.wav"

# The same mixture at 16 kHz, for the 2SPK 16K separation model. Same recipe as
# mix_8k so the two rates differ only in bandwidth; the SS parity cases need an
# overlapping mixture at each model's own rate, from a CC0 source.
ff -i "$TMP/caro_davy.wav" -i "$TMP/bill_boerst.wav" \
    -filter_complex \
    '[0:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=1dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05[a];[1:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=2dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=250[b];[a][b]amix=inputs=2:duration=longest:normalize=0,volume=1/2,volume=2dB,alimiter=limit=0.8:level=false[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 16000 -ac 1 -c:a pcm_s16le "$FA/mix_16k.wav"

# Three-speaker overlap for the 3SPK separation model. Peter starts at 1 s;
# all three speakers overlap for roughly 4.75 s.
ff -i "$TMP/caro_davy.wav" -i "$TMP/bill_boerst.wav" \
    -i "$TMP/peter_yearsley.wav" \
    -filter_complex \
    '[0:a]aresample=8000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=1dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05[a];[1:a]aresample=8000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=2dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=250[b];[2:a]aresample=8000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:5.75,asetpts=PTS-STARTPTS,volume=5dB,afade=t=in:st=0:d=0.05,afade=t=out:st=5.70:d=0.05,adelay=1000[c];[a][b][c]amix=inputs=3:duration=longest:normalize=0,volume=1/3,volume=3dB,alimiter=limit=0.8:level=false[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 8000 -ac 1 -c:a pcm_s16le "$FA/mix3_8k.wav"

# Stereo form of the dialogue, with the two speakers spatially separated.
ff -i "$TMP/caro_davy.wav" -i "$TMP/bill_boerst.wav" \
    -filter_complex \
    '[0:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=1dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=500,pan=stereo|c0=0.85*c0|c1=0.30*c0[a];[1:a]aresample=16000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=2dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=8750,pan=stereo|c0=0.30*c0|c1=0.85*c0[b];[a][b]amix=inputs=2:duration=longest:normalize=0,volume=3dB,alimiter=limit=0.8:level=false,apad=pad_dur=0.75,atrim=0:17.5[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 16000 -ac 2 -c:a pcm_s16le "$FA/multi_speaker.wav"

# Longer three-speaker sequence for streaming and pipeline integration tests.
ff -i "$TMP/caro_davy.wav" -i "$TMP/bill_boerst.wav" \
    -i "$TMP/peter_yearsley.wav" \
    -filter_complex \
    '[0:a]aresample=22050,aformat=sample_fmts=fltp:channel_layouts=mono,volume=1dB,afade=t=in:st=0:d=0.05,afade=t=out:st=8.378:d=0.05,adelay=500[a];[1:a]aresample=22050,aformat=sample_fmts=fltp:channel_layouts=mono,volume=2dB,afade=t=in:st=0:d=0.05,afade=t=out:st=10.782:d=0.05,adelay=8600[b];[2:a]aresample=22050,aformat=sample_fmts=fltp:channel_layouts=mono,volume=5dB,afade=t=in:st=0:d=0.05,afade=t=out:st=5.895:d=0.05,adelay=19100[c];[a][b][c]amix=inputs=3:duration=longest:normalize=0,alimiter=limit=0.8:level=false,apad=pad_dur=0.5,atrim=0:25.55[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 22050 -ac 1 -c:a pcm_s16le "$FA/speech_long.wav"

# Three seconds at S3Gen's own 24 kHz, for the Chatterbox conditioning case that
# deliberately leaves the resampler idle.
ff -i "$TMP/caro_davy.wav" \
    -filter_complex \
    '[0:a]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:3,asetpts=PTS-STARTPTS,volume=1dB,afade=t=in:st=0:d=0.05,afade=t=out:st=2.95:d=0.05[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 24000 -ac 1 -c:a pcm_s16le "$FA/speech_24k.wav"

echo "Deriving CC0 music fixture..."
ff -i "$TMP/music.mp3" -ss 20 -t 10 -ar 44100 -ac 2 \
    -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -c:a pcm_s16le "$FA/music.wav"

# Speech over music gives VAD and vocal-separation tests a licensed mixed-media
# input without pretending the instrumental source contains singing.
ff -i "$TMP/music.mp3" -i "$TMP/caro_davy.wav" \
    -filter_complex \
    '[0:a]atrim=start=20:end=30,asetpts=PTS-STARTPTS,aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=-9dB[m];[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=mono,atrim=0:8,asetpts=PTS-STARTPTS,volume=3dB,afade=t=in:st=0:d=0.05,afade=t=out:st=7.95:d=0.05,adelay=1000,pan=stereo|c0=0.70*c0|c1=0.70*c0[v];[m][v]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.8:level=false[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 44100 -ac 2 -c:a pcm_s16le "$FA/speech_music.wav"

# USS runs at 32 kHz and its parity case needs a window holding both speech and
# music, since it separates by AudioSet condition. Derived from the fixture above
# rather than re-mixed, so the two stay the same material.
ff -i "$FA/speech_music.wav" \
    -filter_complex '[0:a]aresample=32000,aformat=sample_fmts=fltp:channel_layouts=mono[out]' \
    -map '[out]' -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -ar 32000 -ac 1 -c:a pcm_s16le "$FA/speech_music_32k.wav"

echo "Done:"
du -sh "$FA" "$US"
