#!/usr/bin/env python3
"""Check `ModelCatalog`'s `sizeBytes` against what each variant would actually download.

The catalog's sizes were round guesses until 2026-08-12, and a guess that is never
re-checked drifts: super-resolution was declared at 180 MB against a real 439 MB,
because the number had been copied from the entry above it. That figure is shown to
a user before a download, so being wrong by 2.4x is a user-visible bug rather than
untidy bookkeeping.

This resolves every variant's own `files` globs against the Hub's blob listing and
sums what matches. Sizes are read out of ModelCatalog.swift, so the script checks the
committed source rather than a copy of it.

    python3 Scripts/catalog-sizes.py            # report, exit 1 on any mismatch
    python3 Scripts/catalog-sizes.py --tolerance 0.02

Requires network. Private or gated repositories need a token in
~/.cache/huggingface/token; unauthenticated requests conflate private with absent.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG = ROOT / "Sources" / "AudioToolCore" / "Services" / "ModelCatalog.swift"

# Each variant's download manifest, expanded from the `ModelFiles` helper it uses.
# Keep in step with ModelCatalog.swift and ModelRepositories.swift - the point of the
# script is to check the sizes, so the globs are stated here rather than parsed.
STANDARD = lambda p: [f"model_{p}.safetensors", "config.json"]
GAN_PACKAGE = lambda name: [f"{name}/**"]
CHATTERBOX = [
    "model.safetensors",
    "grapheme_mtl_merged_expanded_v1.json",
    "config.json",
    "conds.safetensors",
    "s3tokenizer.safetensors",
    "s3_tokenizer.safetensors",
    "s3tokenizer/model.safetensors",
    "s3_tokenizer/model.safetensors",
]

VARIANTS: list[tuple[str, str, list[str]]] = [
    ("mossformer2_se_fp32", "starkdmi/MossFormer2_SE_48K_MLX", STANDARD("fp32")),
    ("mossformer2_se_fp16", "starkdmi/MossFormer2_SE_48K_MLX", STANDARD("fp16")),
    ("frcrn_se_fp32", "starkdmi/FRCRN_SE_16K_MLX", STANDARD("fp32")),
    ("mossformer_gan_se_coreml_fp16", "starkdmi/MossFormer_GAN_SE_16K_CoreML",
     GAN_PACKAGE("MossFormerGAN_256frames_FP16.mlpackage")),
    ("mossformer_gan_se_coreml_fp32", "starkdmi/MossFormer_GAN_SE_16K_CoreML",
     GAN_PACKAGE("MossFormerGAN_256frames.mlpackage")),
    ("mossformer2_ss_2spk_16k_fp32", "starkdmi/MossFormer2_SS_2SPK_16K_MLX", STANDARD("fp32")),
    ("mossformer2_ss_3spk_8k_fp32", "starkdmi/MossFormer2_SS_3SPK_8K_MLX", STANDARD("fp32")),
    ("mossformer2_ss_2spk_whamr_8k_fp32", "starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX", STANDARD("fp32")),
    ("mossformer2_sr_fp32", "starkdmi/MossFormer2_SR_48K_MLX", STANDARD("fp32")),
    ("mossformer2_sr_int8", "starkdmi/MossFormer2_SR_48K_MLX", STANDARD("int8")),
    ("uss_fp32", "starkdmi/USS_MLX", ["resunet30_fp32.safetensors"]),
    ("uss_fp16", "starkdmi/USS_MLX", ["resunet30_fp16.safetensors"]),
    ("demucs_fp32", "starkdmi/Demucs_MLX",
     ["drums.safetensors", "bass.safetensors", "other.safetensors", "vocals.safetensors"]),
    ("kokoro_tts_bf16", "mlx-community/Kokoro-82M-bf16",
     ["*.safetensors", "voices/*.npy", "config.json"]),
    ("chatterbox_tts_fp32", "starkdmi/chatterbox", CHATTERBOX),
    ("whisper_large_v3", "mlx-community/whisper-large-v3-mlx", ["weights.npz", "config.json"]),
    ("whisper_small", "mlx-community/whisper-small-mlx", ["weights.npz", "config.json"]),
    ("silero_vad_coreml", "FluidInference/silero-vad-coreml",
     ["silero-vad-unified-256ms-v6.2.1.mlmodelc/**"]),
]


def token() -> str | None:
    path = pathlib.Path.home() / ".cache" / "huggingface" / "token"
    return path.read_text().strip() if path.exists() else None


def blobs(repo: str, auth: str | None) -> dict[str, int]:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}?blobs=true",
        headers={"Authorization": f"Bearer {auth}"} if auth else {},
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    return {f["rfilename"]: (f.get("size") or 0) for f in payload.get("siblings", [])}


def matcher(glob: str) -> re.Pattern[str]:
    """`**` crosses path separators, `*` does not - the same rule the Swift side uses."""
    out, i = "", 0
    while i < len(glob):
        if glob.startswith("**", i):
            out, i = out + ".*", i + 2
        elif glob[i] == "*":
            out, i = out + "[^/]*", i + 1
        else:
            out, i = out + re.escape(glob[i]), i + 1
    return re.compile(f"^{out}$")


def declared_sizes() -> dict[str, int]:
    """Read each variant's `sizeBytes` out of the catalog source."""
    source = CATALOG.read_text()
    found: dict[str, int] = {}
    for match in re.finditer(r'^\s*id: "([a-z0-9_]+)",\s*$', source, re.MULTILINE):
        tail = source[match.end():match.end() + 400]
        size = re.search(r"sizeBytes:\s*([0-9_]+(?:\s*\*\s*[0-9_]+)?)", tail)
        if size:
            found[match.group(1)] = eval(size.group(1).replace("_", ""))  # ints only
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="fractional difference to accept (default: exact)")
    args = parser.parse_args()

    auth, declared, cache = token(), declared_sizes(), {}
    failures = 0

    print(f"{'variant':36s} {'declared':>14s} {'actual':>14s}  status")
    for name, repo, globs in VARIANTS:
        if repo not in cache:
            try:
                cache[repo] = blobs(repo, auth)
            except urllib.error.HTTPError as error:
                print(f"{name:36s} {'':>14s} {'':>14s}  UNREACHABLE {repo} ({error.code})")
                failures += 1
                continue

        patterns = [matcher(g) for g in globs]
        matched = {f: s for f, s in cache[repo].items() if any(p.match(f) for p in patterns)}
        actual = sum(matched.values())
        stated = declared.get(name)

        if stated is None:
            status = "NOT IN CATALOG"
        elif not matched:
            status = "MANIFEST MATCHES NOTHING"
        elif stated == actual:
            status = "ok"
        elif args.tolerance and abs(actual - stated) <= args.tolerance * actual:
            status = f"ok (within {args.tolerance:.0%})"
        else:
            status = f"MISMATCH {actual / stated:.2f}x"

        if not status.startswith("ok"):
            failures += 1
        print(f"{name:36s} {stated or 0:>14,} {actual:>14,}  {status}")

    if failures:
        print(f"\n{failures} variant(s) need attention.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
