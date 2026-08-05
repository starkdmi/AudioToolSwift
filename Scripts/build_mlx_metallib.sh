#!/usr/bin/env bash
#
# Build MLX's Metal shader library so `swift test` and `swift run` work.
#
# Why this exists
# ---------------
# SwiftPM cannot compile .metal sources. mlx-swift ships 32 Metal kernels under
# Source/Cmlx/mlx/mlx/backend/metal/kernels but its Package.swift declares no
# resource, plugin or binary target that turns them into a .metallib - that step
# only happens in Xcode's build system. So `swift build` succeeds (the C++ and
# Swift compile fine) and then every Metal operation fails at runtime with
# "Failed to load the default metallib", because the library was never produced.
#
# This is upstream and unresolved: ml-explore/mlx-swift#349 is open, unassigned
# and has no maintainer response, and its reporter proposes exactly what is
# missing - that default.metallib be declared as a SwiftPM resource.
#
# How the fix works
# -----------------
# MLX looks for its shader library in four places, in order (see
# Source/Cmlx/mlx/mlx/backend/metal/device.cpp, load_default_library):
#
#   1. mlx.metallib next to the running binary        <- what we target
#   2. Resources/mlx.metallib next to the binary
#   3. default.metallib inside an mlx-swift_Cmlx.bundle
#   4. a compiled-in default path
#
# The first is by far the easiest to satisfy from the command line: compile the
# kernels ourselves and drop mlx.metallib beside the test binary.
#
# Approach follows github.com/soniqo/speech-swift, which solves the same problem
# for a comparable MLX + CoreML speech stack.
#
# Usage:  Scripts/build_mlx_metallib.sh [debug|release] [--force]
#
# Requires the Metal Toolchain, which ships separately from Xcode:
#   xcodebuild -downloadComponent MetalToolchain
#
set -euo pipefail

CONFIG="debug"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    debug|release) CONFIG="$arg" ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/.build}"
OUT_DIR="$BUILD_DIR/$CONFIG"
MLX_DIR="$BUILD_DIR/checkouts/mlx-swift"
KERNELS="$MLX_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"

[[ -d "$BUILD_DIR" ]] || { echo "error: $BUILD_DIR not found - run 'swift build' first" >&2; exit 1; }
[[ -d "$OUT_DIR"   ]] || { echo "error: $OUT_DIR not found - run 'swift build' for config '$CONFIG'" >&2; exit 1; }
[[ -d "$KERNELS"   ]] || { echo "error: MLX kernels not found at $KERNELS" >&2; exit 1; }

if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  echo "error: the Metal Toolchain is not installed." >&2
  echo "       run: xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi

OUT_METALLIB="$OUT_DIR/mlx.metallib"
HASH_FILE="$OUT_DIR/.mlx.metallib.sha"

# Skip the ~30s recompile when the kernel sources have not changed. Hash headers
# too: a .h edit changes the output without touching any .metal file.
CURRENT_HASH="$(find "$KERNELS" -type f \( -name '*.metal' -o -name '*.h' \) ! -name '*_nax.metal' \
  | LC_ALL=C sort | xargs cat | shasum -a 256 | awk '{print $1}')"

NEEDS_BUILD=1
if [[ "$FORCE" != "1" && -f "$OUT_METALLIB" && -f "$HASH_FILE" ]]; then
  [[ "$CURRENT_HASH" == "$(cat "$HASH_FILE")" ]] && NEEDS_BUILD=0
fi

if [[ "$NEEDS_BUILD" == "1" ]]; then
  SRCS=()
  while IFS= read -r line; do SRCS+=("$line"); done \
    < <(find "$KERNELS" -type f -name '*.metal' ! -name '*_nax.metal' | LC_ALL=C sort)
  [[ "${#SRCS[@]}" -gt 0 ]] || { echo "error: no .metal sources under $KERNELS" >&2; exit 1; }

  TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlx-metallib.XXXXXX")"
  trap 'rm -rf "$TMP"' EXIT

  echo "Compiling ${#SRCS[@]} Metal kernels ($CONFIG)..."
  AIR=()
  for SRC in "${SRCS[@]}"; do
    # Flatten the relative path into a unique name - several kernels share a
    # basename across subdirectories and would otherwise overwrite each other.
    KEY="$(printf '%s' "${SRC#"$KERNELS/"}" | shasum -a 256 | awk '{print $1}' | cut -c1-16)"
    xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math \
      -Wno-c++17-extensions -Wno-c++20-extensions \
      -c "$SRC" -I"$KERNELS" -I"$MLX_DIR/Source/Cmlx/mlx" -o "$TMP/$KEY.air"
    AIR+=("$TMP/$KEY.air")
  done

  echo "Linking $OUT_METALLIB"
  xcrun -sdk macosx metallib "${AIR[@]}" -o "$OUT_METALLIB"
  printf '%s' "$CURRENT_HASH" > "$HASH_FILE"
else
  echo "mlx.metallib is up to date, skipping compile"
fi

# Place it beside every test binary; search strategy 1 is relative to the
# executable, and the xctest bundle's binary lives in Contents/MacOS.
ARCH="$(uname -m)-apple-macosx"
FOUND=0
for CFG in debug release; do
  for BUNDLE in "$BUILD_DIR/$ARCH/$CFG"/*.xctest; do
    [[ -d "$BUNDLE/Contents/MacOS" ]] || continue
    cp -f "$OUT_METALLIB" "$BUNDLE/Contents/MacOS/mlx.metallib"
    echo "OK: $BUNDLE/Contents/MacOS/mlx.metallib"
    FOUND=1
  done
done
[[ "$FOUND" == "1" ]] || echo "note: no .xctest bundles yet - run 'swift build --build-tests' then re-run this"

echo "Done."
