#!/usr/bin/env bash
#
# Fetch and derive the test audio fixtures.
#
# Everything here is openly licensed and small. The upstream speech samples come
# from ClearerVoice-Studio, which is Apache-2.0 - the same licence as this project -
# and which is where MossFormer2 SE/SS/SR and FRCRN originate, so these are the
# clips those models are demoed on. Music is CC0.
#
# The committed fixtures were produced by this script; it exists so their
# provenance is reproducible rather than asserted.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CVS="https://raw.githubusercontent.com/modelscope/ClearerVoice-Studio/main"
FA="Tests/AudioToolFluidAudioTests/Fixtures"
US="Tests/AudioToolUSSTests/Fixtures"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$FA" "$US"

command -v ffmpeg >/dev/null || { echo "error: ffmpeg required (brew install ffmpeg)" >&2; exit 1; }

echo "Fetching ClearerVoice-Studio samples (Apache-2.0)..."
curl -sfL "$CVS/clearvoice/samples/path_to_input_wavs_sr/LJ001-0001.wav" -o "$TMP/ljspeech.wav"
curl -sfL "$CVS/clearvoice/samples/input_ss.wav"                          -o "$TMP/dialogue.wav"
curl -sfL "$CVS/clearvoice/samples/path_to_input_wavs_ss/speech_mixure1.wav" -o "$TMP/multi.wav"
curl -sfL "$CVS/clearvoice/samples/path_to_input_wavs_ss/050a0501_1.7783_442o030z_-1.7783.wav" -o "$TMP/mix2.wav"
curl -sfL "$CVS/clearvoice/samples/test.wav"                              -o "$TMP/test48.wav"
curl -sfL "$CVS/speechscore/audios/clean.wav"                             -o "$TMP/clean.wav"
curl -sfL "$CVS/clearvoice/samples/input.wav"                             -o "$TMP/enhance.wav"

echo "Fetching CC0 music (archive.org, CC0 1.0)..."
curl -sfL "https://archive.org/download/MarchForHonor/Distant_Wonders.mp3" -o "$TMP/music.mp3"

echo "Deriving fixtures..."
cp "$TMP/ljspeech.wav" "$FA/speech_long.wav"
cp "$TMP/dialogue.wav" "$FA/speech_dialogue.wav"
cp "$TMP/dialogue.wav" "$US/speech_dialogue.wav"
cp "$TMP/multi.wav"    "$FA/multi_speaker.wav"
# test.wav must be speech at 16 kHz: it backs the VAD, transcription and
# diarization tests. ClearerVoice's own test.wav is 48 kHz and not speech-dominant,
# so this concatenates three of their speech samples instead.

ff() { ffmpeg -hide_banner -loglevel error -y "$@"; }
ff -i "$TMP/enhance.wav" -i "$TMP/clean.wav" -i "$TMP/dialogue.wav" \
   -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1" \
   -ar 16000 -ac 1 -c:a pcm_s16le "$FA/test.wav"
cp "$FA/test.wav" "$US/test.wav"
ff -i "$TMP/test48.wav" -ar 48000 -ac 1 -c:a pcm_s16le "$FA/test_48k.wav"
ff -i "$TMP/music.mp3" -ss 20 -t 10 -ar 44100 -ac 2 -c:a pcm_s16le "$FA/music.wav"
ff -i "$TMP/mix2.wav"  -ar 8000  -ac 1 -c:a pcm_s16le "$FA/mix_8k.wav"
ff -i "$TMP/ljspeech.wav" -ar 48000 -ac 1 -c:a pcm_s16le "$FA/speech_48k.wav"
# Three overlapping talkers, staggered, for 3-speaker separation
ff -i "$TMP/dialogue.wav" -i "$TMP/clean.wav" -i "$TMP/enhance.wav" \
   -filter_complex "[0:a]atrim=0:4,asetpts=PTS-STARTPTS,volume=0.5[a];\
[1:a]atrim=0:4,asetpts=PTS-STARTPTS,adelay=300|300,volume=0.5[b];\
[2:a]atrim=0:4,asetpts=PTS-STARTPTS,adelay=900|900,volume=0.5[c];\
[a][b][c]amix=inputs=3:normalize=0" \
   -ar 8000 -ac 1 -c:a pcm_s16le "$FA/mix3_8k.wav"

echo "Done:"; du -sh "$FA" "$US"
