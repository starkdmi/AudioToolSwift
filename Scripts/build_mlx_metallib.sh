#!/usr/bin/env bash
#
# Build MLX's Metal shader library so `swift test` and `swift run` work.
#
# Why this exists
# ---------------
# Bare command-line SwiftPM cannot compile MLX's .metal sources. mlx-swift ships
# generated kernels under Source/Cmlx/mlx-generated/metal; Xcode's build system
# turns them into default.metallib and bundles it automatically, while
# `swift build`, `swift test` and `swift run` do not. The Swift/C++ build can
# therefore succeed and the first Metal operation can still abort at runtime.
#
# This is documented by mlx-swift itself:
#   github.com/ml-explore/mlx-swift#swiftpm
#   github.com/ml-explore/mlx-swift/issues/36
#
# It is inherited by mlx-swift-lm because that package runs on mlx-swift. It does
# NOT require an extra step in an Xcode or xcodebuild workflow. This script is
# only a local workaround for intentionally using bare SwiftPM (notably CI).
# mlx-swift#349 is a separate Tuist resource-bundle issue and is not the basis
# for this workaround.
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
# Use the pre-generated, self-contained kernel set - this is what mlx-swift's
# SwiftPM build compiles. The other tree, mlx/mlx/backend/metal/kernels, holds 32
# kernels that need version-gated includes (metal_3_0 vs metal_3_1) and a curated
# per-kernel build list; compiling it by globbing fails on a missing bf16.h.
KERNELS="$MLX_DIR/Source/Cmlx/mlx-generated/metal"

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
CURRENT_HASH="$(find "$KERNELS" -type f \( -name '*.metal' -o -name '*.h' \) \
  | LC_ALL=C sort | xargs cat | shasum -a 256 | awk '{print $1}')"

NEEDS_BUILD=1
if [[ "$FORCE" != "1" && -f "$OUT_METALLIB" && -f "$HASH_FILE" ]]; then
  [[ "$CURRENT_HASH" == "$(cat "$HASH_FILE")" ]] && NEEDS_BUILD=0
fi

if [[ "$NEEDS_BUILD" == "1" ]]; then
  SRCS=()
  while IFS= read -r line; do SRCS+=("$line"); done \
    < <(find "$KERNELS" -type f -name '*.metal' | LC_ALL=C sort)
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
      -c "$SRC" -I"$KERNELS" -o "$TMP/$KEY.air"
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
