#!/usr/bin/env python3
"""
What each precision of a model costs, in one table.

Parity answers "does Swift compute what Python computes at this precision". It
deliberately cannot answer "should I ship int4", because both sides quantize
identically and the error cancels - a passing parity run says nothing about
quality. This says what quantization costs, so a caller can choose.

Three columns come from three places, and none of them substitutes for another:

  size      the checkpoint on disk
  speed     `audio-tool-bench` report JSON - RTF and peak footprint
  quality   the parity artifacts, quantized `enhanced` against fp32 `enhanced`

Quality is measured between the *reference* outputs rather than by running Swift
again. That is not a shortcut: the parity suite holds Swift to within ~77 dB of
these same references, and the quantization loss being measured is 39-59 dB, so
the Swift and Python figures agree to well inside the last digit reported here.
Re-running Swift would produce the same table an hour later.

Usage:
    Scripts/quantization-report.py --bench BenchmarkResults/bench-*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import textwrap
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "Parity" / "artifacts"
WEIGHTS = ROOT / "Parity" / "weights"

# Model -> (benchmark id prefix, artifact stem, weights directory, tensor,
#           [(precision, weights filename)]). fp32 first: it is the reference every
#           other row is measured against.
#
# The weights directory is relative to the package for staged repositories and to
# the research checkout for models that were never published - USS among them, so
# its checkpoints exist only in `Models/`.
MODELS = {
    "mossformer2_se_48k": (
        "mlx.mossformer2_se_48k",
        "mossformer2_se_48k_direct",
        WEIGHTS / "MossFormer2_SE_48K_MLX",
        "enhanced",
        [
            ("fp32", "model_fp32.safetensors"),
            ("fp16", "model_fp16.safetensors"),
            ("int8", "model_int8.safetensors"),
            ("int6", "model_int6.safetensors"),
            ("int4", "model_int4.safetensors"),
        ],
    ),
    # fp16 is absent on purpose: the conversion returns NaN for every sample while
    # its checkpoint holds no non-finite weight, so the forward pass overflows. It
    # would render as a row of dashes and read like missing data rather than a
    # result. `MossFormer2SR48KProvider.supportedPrecisions` is where that lives.
    #
    # int6 and int4 stay here even though that same list no longer offers them and
    # neither is on the Hub. This table is the record of what quantizing this model
    # costs, and dropping the two rows that make the point - bigger than int8 *and*
    # worse, at identical speed and memory - would leave the recommendation above
    # them unevidenced. `UNPUBLISHED` marks them so neither table reads as an offer.
    "mossformer2_sr_48k": (
        "mlx.mossformer2_sr_48k",
        "mossformer2_sr_48k_direct",
        WEIGHTS / "MossFormer2_SR_48K_MLX",
        "enhanced",
        [
            ("fp32", "model_fp32.safetensors"),
            ("int8", "model_int8.safetensors"),
            ("int6", "model_int6.safetensors"),
            ("int4", "model_int4.safetensors"),
        ],
    ),
    # Two precisions, no quantization - `nn.quantize` appears nowhere in the USS
    # tree in any language. Included because "which precision should I ship" is the
    # same question whether the answer is a dtype or a bit width, and here it has an
    # unusually clear answer.
    #
    # `separated_music` is the harder of the two conditions and the one reported;
    # speech measures ~11 dB better on the same run.
    "uss_resunet30_32k": (
        "mlx.uss",
        "uss_resunet30_32k",
        ROOT.parent / "Models" / "uss_mlx_swift" / "USSSwift" / "Models",
        "separated_music",
        [
            ("fp32", "resunet30_fp32.safetensors"),
            ("fp16", "resunet30_fp16.safetensors"),
        ],
    ),
}


# Checkpoint sizes in bytes, recorded when they were measured.
#
# Used only when the file is not on disk. Several of these were deleted once they
# had been measured - a 64 MiB checkpoint that is measurably worse than the fp16
# beside it is not worth keeping on a full disk - and without this the table would
# render a blank where the reason for deleting it used to be. The measurement is
# the deliverable; the weights were the means.
KNOWN_SIZES: dict[str, dict[str, int]] = {
    "mossformer2_se_48k": {
        "model_fp32.safetensors": 221_178_088,
        "model_fp16.safetensors": 110_652_628,
        "model_int8.safetensors": 90_089_394,
        "model_int6.safetensors": 78_686_048,
        "model_int4.safetensors": 67_282_702,
    },
    "mossformer2_sr_48k": {
        "model_fp32.safetensors": 438_668_396,
        "model_fp16.safetensors": 219_412_855,
        "model_int8.safetensors": 307_581_429,
        "model_int6.safetensors": 296_178_117,
        "model_int4.safetensors": 284_774_853,
    },
}


# Precisions that were measured but are not on the Hub, and why.
#
# Not the same thing as a verdict, which is why it is a separate table: "avoid" is
# advice about a checkpoint a reader can still download, and MossFormer2 SE
# publishes int6 and int4 despite both carrying that mark. These rows have no file
# behind them at all, so a reader treating one as an option is looking for
# something that 404s.
#
# The rows stay in the table regardless. The comparison is the deliverable - it is
# what shows that quantizing SR buys size and nothing else - and deleting the row
# would delete the evidence for the recommendation above it.
UNPUBLISHED: dict[str, dict[str, str]] = {
    "mossformer2_sr_48k": {
        "fp16": "all-NaN forward pass",
        "int6": "strictly dominated by int8",
        "int4": "strictly dominated by int8",
    },
}


# Verdict per precision: (mark, one-line reason).
#
# A judgement, not a measurement, which is why it is declared here rather than
# derived from the numbers - "0.30x the size for 39 dB" is a fact, and whether that
# is a good trade depends on what you are shipping. Kept beside the generator so
# the tables cannot drift from the recommendation the way a hand-maintained
# markdown file would.
#
#   good        the one to reach for
#   ok          defensible under a specific constraint, named in the reason
#   avoid       measurably worse than an alternative at no saving that justifies it
#   broken      does not produce usable output
VERDICTS: dict[str, dict[str, tuple[str, str]]] = {
    "mossformer2_se_48k": {
        "fp32": ("ok", "reference quality, and only 16% slower than fp16"),
        "fp16": ("good", "fastest, half the size, 70 dB"),
        "int8": ("ok", "0.41x the download for 59 dB - no faster than fp32, and more memory"),
        "int6": ("avoid", "slower than fp16, 22 dB worse, saves 31 MiB"),
        "int4": ("avoid", "slower than fp16, 31 dB worse, saves 42 MiB"),
    },
    "mossformer2_sr_48k": {
        "fp32": ("good", "quantization changes neither speed nor memory here"),
        "int8": ("ok", "30% smaller for 62 dB, if download size is the constraint"),
        "int6": ("avoid", "no speed or memory gain, 51 dB"),
        "int4": ("avoid", "no speed or memory gain, 38 dB"),
        "fp16": ("broken", "returns NaN for every sample - the forward pass overflows"),
    },
    "uss_resunet30_32k": {
        "fp32": ("good", "faster and lower memory than fp16"),
        "fp16": ("ok", "half the download, but slower, more memory and 57 dB"),
    },
}


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB.

    Scale-invariant because a precision change can shift overall gain slightly
    without that being audible degradation, and a plain SNR would charge for it.
    """
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    projection = (np.dot(estimate, reference) / np.dot(reference, reference)) * reference
    residual = estimate - projection
    return float(10 * np.log10(np.dot(projection, projection) / np.dot(residual, residual)))


def log_spectral_distance(estimate: np.ndarray, reference: np.ndarray, n: int = 1024) -> float:
    """Mean per-frame log-spectral distance.

    Carried alongside SI-SDR because the two fail differently: SI-SDR is dominated
    by whatever is loudest, so a quantization that wrecks a quiet high band can
    still score well. LSD weights every band equally and catches it.
    """
    window = np.hanning(n)

    def spectrum(x: np.ndarray) -> np.ndarray:
        frames = [np.abs(np.fft.rfft(x[i : i + n] * window)) for i in range(0, len(x) - n, n // 2)]
        return np.log10(np.asarray(frames) ** 2 + 1e-10)

    return float(np.mean(np.sqrt(np.mean((spectrum(estimate) - spectrum(reference)) ** 2, axis=1))))


def bench_rows(reports: list[Path], prefix: str) -> dict[str, dict]:
    """Speed and memory per precision, keyed by the id suffix.

    Several reports, not one. The runner writes a file per invocation, and the
    memory numbers are only trustworthy when a case had the machine to itself - so
    a precision ladder is usually several `-c` runs rather than one `-f` sweep, and
    the rows for one model arrive spread across files. Later files win, which makes
    re-measuring a single precision a matter of running it again.
    """
    rows: dict[str, dict] = {}
    results = [r for report in reports for r in json.loads(report.read_text())["results"]]
    for r in results:
        if not r["id"].startswith(prefix) or r.get("status") != "completed":
            continue
        rows[r["id"].rsplit(".", 1)[-1]] = {
            "rtf": r["timing"]["realTimeFactor"],
            "median": r["timing"]["medianSeconds"],
            "load": r["timing"]["loadSeconds"],
            "peak": r["memory"]["peakFootprintBytes"],
        }
    return rows


def report_for(model: str, bench: list[Path], public: bool = False) -> str:
    prefix, stem, weights_dir, tensor, precisions = MODELS[model]
    speed = bench_rows(bench, prefix)

    def artifact(precision: str) -> np.ndarray | None:
        # fp32 is the unsuffixed case - it predates the precision ladder and other
        # things reference it by that name.
        name = stem if precision == "fp32" else f"{stem}_{precision}"
        path = ARTIFACTS / f"{name}.safetensors"
        if not path.exists():
            return None
        return load_file(path)[tensor].astype(np.float64)

    reference = artifact("fp32")
    if reference is None:
        raise SystemExit(f"{model}: no fp32 artifact to measure against - run Parity/generate.py")

    marks = VERDICTS.get(model, {})
    unpublished = UNPUBLISHED.get(model, {})
    if public:
        lines = [
            "| precision | size | speed | memory | quality vs fp32 | |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| precision | size | vs fp32 | RTF | peak | SI-SDR vs fp32 | LSD | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    base_path = weights_dir / precisions[0][1]
    base_size = (
        base_path.stat().st_size if base_path.exists()
        else KNOWN_SIZES.get(model, {}).get(precisions[0][1], 0)
    )

    for precision, filename in precisions:
        path = weights_dir / filename
        size = (
            path.stat().st_size if path.exists()
            else KNOWN_SIZES.get(model, {}).get(filename)
        )
        measured = artifact(precision)
        row = speed.get(precision, {})

        size_text = f"{size / 2**20:.0f} MiB" if size else "-"
        ratio = f"{size / base_size:.2f}x" if size and base_size else "-"
        rtf = f"{row['rtf']:.1f}x" if row else "-"
        peak = f"{row['peak'] / 2**20:.0f} MiB" if row else "-"
        if precision == "fp32":
            quality, spectral = "reference", "-"
        elif measured is None:
            quality, spectral = "-", "-"
        else:
            n = min(len(measured), len(reference))
            quality = f"{si_sdr(measured[:n], reference[:n]):.1f} dB"
            spectral = f"{log_spectral_distance(measured[:n], reference[:n]):.3f}"

        mark, reason = marks.get(precision, ("", ""))
        unavailable = unpublished.get(precision)
        if public:
            # "not published" outranks the verdict badge: whether a reader should
            # want it is moot when there is nothing to download.
            badge = {"good": "**recommended**", "ok": "situational",
                     "avoid": "not recommended", "broken": "**do not use**"}.get(mark, "")
            if unavailable:
                badge = "not published"
            note = f"{badge} - {reason}" if reason else badge
            lines.append(f"| {precision} | {size_text} | {rtf} | {peak} | {quality} | {note} |")
        else:
            tag = f"{mark}, unpublished" if unavailable else mark
            lines.append(
                f"| {precision} | {size_text} | {ratio} | {rtf} | {peak} | {quality} "
                f"| {spectral} | {tag} |"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", help="audio-tool-bench report JSON (glob accepted)")
    parser.add_argument("--model", default="mossformer2_se_48k", choices=sorted(MODELS))
    parser.add_argument(
        "--public", action="store_true",
        help="render for a model card: plain wording, no LSD or size ratio, "
             "verdicts spelled out rather than tagged",
    )
    args = parser.parse_args()

    bench: list[Path] = []
    if args.bench:
        matches = sorted(glob.glob(args.bench))
        if not matches:
            raise SystemExit(f"no benchmark report matched {args.bench}")
        bench = [Path(m) for m in matches]

    print(f"\n## {args.model}\n")
    print(report_for(args.model, bench, public=args.public))

    # Only the precisions this model actually renders. `UNPUBLISHED` records
    # availability for every precision that was measured, including ones no table
    # shows - SR's fp16 is unpublished *and* omitted from `MODELS`, and a note
    # explaining a row the reader cannot see is worse than no note.
    rendered = [p for p, _ in MODELS[args.model][4]]
    absent = {p: why for p, why in UNPUBLISHED.get(args.model, {}).items() if p in rendered}
    if absent:
        published = [p for p in rendered if p not in absent]
        # Named individually rather than counted: a reader deciding what to download
        # needs to know which rows are measurements and which are options.
        reasons = "; ".join(f"{p} ({why})" for p, why in absent.items())
        # Wrapped rather than printed as one long line, to match the fixed notes
        # below and keep the generated markdown diffable.
        print("\n" + textwrap.fill(
            f"**Only {' and '.join(published)} are published.** The remaining rows "
            f"are measurements rather than downloads - {reasons}. They are kept "
            f"because the comparison is the point: it is what shows what quantizing "
            f"this model does and does not buy.",
            width=76,
        ))

    if args.public:
        # No report filename and no host: a model card is read by people who do not
        # have this machine, and a run identifier they cannot reproduce reads as
        # provenance while providing none.
        print("\nMeasured on an Apple M1 Pro (16 GB) over 30 s of audio, three timed")
        print("runs per configuration. Speed is realtime factor - higher is faster.")
        print("Expect different absolute numbers on other hardware; the ordering has")
        print("been stable across the models measured so far, but is not guaranteed.")
        print("\nQuality compares each precision against **this model's own fp32")
        print("output**. It is what the precision costs, not how good the model is.")
    else:
        if bench:
            print(f"\nSpeed and memory from {len(bench)} report(s), latest `{bench[-1].name}`.")
        print("\nQuality is measured against this model's own fp32 output, not the clean")
        print("signal - it is the cost of the precision change, not the model's accuracy.")


if __name__ == "__main__":
    main()
