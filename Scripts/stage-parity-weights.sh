#!/usr/bin/env bash
#
# Stage the HuggingFace-cached weights where a test runner can actually read them.
#
# The Swift Hub library caches under ~/Documents/huggingface/models, which macOS
# protects with TCC. A process that has not been granted "Files and Folders >
# Documents" gets NSCocoaErrorDomain 257 - "you don't have permission to view it" -
# and the xcodebuild test runner is such a process. Four parity cases failed on
# exactly that before this existed, while the four using explicit local paths
# passed.
#
# Hardlinks, not copies: same volume, so 1.3 GB of weights costs no extra space,
# and TCC checks the path being opened rather than the inode. Falls back to a copy
# across volumes.
#
# Idempotent. Run again after downloading a new model.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE="${HOME}/Documents/huggingface/models"
DEST="Parity/weights"

REPOS=(
    starkdmi/MossFormer2_SE_48K_MLX
    starkdmi/MossFormer2_SR_48K_MLX
    starkdmi/MossFormer2_SS_2SPK_16K_MLX
    starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX
    starkdmi/MossFormer2_SS_3SPK_8K_MLX
)

[ -d "$CACHE" ] || { echo "error: no HF cache at $CACHE" >&2; exit 1; }

staged=0
missing=0
for repo in "${REPOS[@]}"; do
    src="$CACHE/$repo"
    dst="$DEST/${repo##*/}"
    if [ ! -d "$src" ]; then
        echo "  missing: $repo (download it first)"
        missing=$((missing + 1))
        continue
    fi
    mkdir -p "$dst"
    # Weights and config only. The cache also holds .DS_Store and a .cache
    # directory, neither of which any provider looks at.
    find "$src" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.json' \) \
        ! -name '.*' -print0 |
    while IFS= read -r -d '' file; do
        target="$dst/$(basename "$file")"
        [ -e "$target" ] && continue
        ln "$file" "$target" 2>/dev/null || cp "$file" "$target"
    done
    echo "  staged:  ${repo##*/}"
    staged=$((staged + 1))
done

echo
echo "$staged repo(s) in $DEST${missing:+, $missing missing}"
du -sh "$DEST" 2>/dev/null || true
