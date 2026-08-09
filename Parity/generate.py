#!/usr/bin/env python3
"""
Generate parity artifacts from the reference MLX Python implementations.

    python Parity/generate.py --list
    python Parity/generate.py frcrn_se_16k
    python Parity/generate.py --all

Needs the `mlx` conda environment - `edge-ml` has a torchaudio that will not
load, and every reference runner reads audio through it:

    /opt/homebrew/Caskroom/miniforge/base/envs/mlx/bin/python Parity/generate.py --all

Writes `Parity/artifacts/<case>.safetensors` and `<case>.json`. Those artifacts,
not this script, are what the Swift tests consume.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

import numpy as np

PARITY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PARITY_DIR))

from harness import ARTIFACT_DIR, Context, write  # noqa: E402

# Adapter module -> the case(s) it produces. Kept explicit so `--list` works
# without importing MLX or touching a single weight file.
ADAPTERS = {
    "frcrn": ["frcrn_se_16k"],
    "gan_se": ["mossformer_gan_se_16k_coreml"],
    "mossformer2_se": ["mossformer2_se_48k", "mossformer2_se_48k_direct"],
    "mossformer2_sr": ["mossformer2_sr_48k", "mossformer2_sr_48k_direct"],
    "mossformer2_ss": [
        "mossformer2_ss_2spk_16k", "mossformer2_ss_2spk_16k_direct",
        "mossformer2_ss_2spk_whamr_8k", "mossformer2_ss_2spk_whamr_8k_direct",
        "mossformer2_ss_3spk_8k", "mossformer2_ss_3spk_8k_direct",
    ],
    "demucs": ["demucs_vocals_44k"],
    "uss": ["uss_resunet30_32k"],
    "chatterbox": ["chatterbox_conditionals_24k", "chatterbox_conditionals_22k_long"],
}


def run(adapter_name: str, ctx: Context) -> bool:
    module = importlib.import_module(f"adapters.{adapter_name}")
    print(f"\n=== {adapter_name} ===")
    try:
        built = module.build(ctx)
    except FileNotFoundError as error:
        print(f"  skipped: {error}")
        return True
    except Exception:
        traceback.print_exc()
        print(f"  FAILED: {adapter_name}")
        return False

    # Adapters covering several configurations return a list; one model that
    # ships as three checkpoints should be three cases, not one averaged verdict.
    for case in built if isinstance(built, list) else [built]:
        tensors_path, sidecar_path = write(case)
        print(f"  {case.name}")
        for name, value in case.tensors.items():
            print(f"    {name:20} {tuple(np.shape(value))}")
        size_mb = tensors_path.stat().st_size / 1e6
        print(f"    -> {tensors_path.name} ({size_mb:.1f} MB) + {sidecar_path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="adapter names to run")
    parser.add_argument("--all", action="store_true", help="run every adapter")
    parser.add_argument("--list", action="store_true", help="list adapters and exit")
    args = parser.parse_args()

    if args.list:
        for adapter, cases in ADAPTERS.items():
            print(f"{adapter:20} -> {', '.join(cases)}")
        return 0

    selected = list(ADAPTERS) if args.all else args.cases
    if not selected:
        parser.error("name at least one adapter, or pass --all")

    unknown = [name for name in selected if name not in ADAPTERS]
    if unknown:
        parser.error(f"unknown adapter(s): {', '.join(unknown)}")

    ctx = Context()
    ok = all([run(name, ctx) for name in selected])
    print(f"\nartifacts in {ARTIFACT_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
