#!/usr/bin/env python3
"""Convert MossFormer2 SR 48K fp32 -> fp16 / int8 / int6 / int4.

Mirrors mossformer2_se_mlx/public/python/quantize.py: build the model, run
nn.quantize at group_size 64, save the resulting parameters. fp16 is a plain
astype with no module surgery.

Kept because only fp32 is published, so these checkpoints exist nowhere else and
deleting them to reclaim disk would otherwise lose them. Regenerating takes about
two minutes.

**What it produces is mostly not worth keeping**, which is the reason to keep the
script rather than the output. Measured on an M1 Pro:

  fp16   NaN for every sample. The checkpoint holds no non-finite weight; the
         forward pass overflows. Do not ship it.
  int8   0.70x the size, same RTF, same peak memory, 62.5 dB against fp32
  int6   0.68x, same, same, 50.8 dB
  int4   0.65x, same, same, 38.1 dB

Only 168 `Linear` modules quantize against a Generator that is almost entirely
convolutions, so the file barely shrinks and nothing gets faster. Run this when
you want to re-measure, not to produce something to ship.

Usage:
    Scripts/convert-sr-precisions.py Parity/weights/MossFormer2_SR_48K_MLX
"""
import json, sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

REF = Path("/path/to/clear_voice_research/Models/python/mossformer2_sr_mlx")
sys.path.insert(0, str(REF))
from mossformer2_sr_wrapper import MossFormer2_SR_48K

class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SRC = OUT / "model_fp32.safetensors"
CFG = json.loads((OUT / "config.json").read_text())

GROUP_SIZE = 64

def build():
    args = AttrDict(dict(CFG))
    args.one_time_decode_length = 20.0
    args.decode_window = 4.0
    return MossFormer2_SR_48K(args)

weights = mx.load(str(SRC))
print(f"source: {len(weights)} tensors")

# fp16: dtype only.
fp16 = {k: (v.astype(mx.float16) if v.dtype == mx.float32 else v) for k, v in weights.items()}
mx.save_safetensors(str(OUT / "model_fp16.safetensors"), fp16)
print("wrote model_fp16.safetensors")

for bits in (8, 6, 4):
    model = build()
    model.update(tree_unflatten(list(weights.items())))
    nn.quantize(model, group_size=GROUP_SIZE, bits=bits)
    params = dict(tree_flatten(model.parameters()))
    n_scales = sum(1 for k in params if k.endswith(".scales"))
    mx.save_safetensors(str(OUT / f"model_int{bits}.safetensors"), params)
    print(f"wrote model_int{bits}.safetensors  ({len(params)} tensors, {n_scales} quantized modules)")

    cfg = dict(CFG)
    cfg["quantization_config"] = {"bits": bits, "group_size": GROUP_SIZE}
    (OUT / f"config_int{bits}.json").write_text(json.dumps(cfg, indent=2))
    del model
