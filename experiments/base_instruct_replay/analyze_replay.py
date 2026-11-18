#!/usr/bin/env python3
"""Balanced path-block slope analysis for the Base vs. Instruct replay experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats import (
    descriptive,
    exact_sign_flip_p,
    factorial_contrasts,
    holm_adjust,
    ols_slope,
)

EFFECTS = ("overall", "extraversion", "neuroticism", "interaction")
MODERATORS = EFFECTS[1:]
QUADRANTS = ("high_e_high_n", "high_e_low_n", "low_e_high_n", "low_e_low_n")


def read_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def analyze(
    records: Sequence[dict[str, Any]],
    *,
    primary_axes: Sequence[str] | None = None,
    robustness_axes: Sequence[str] = (),
) -> dict[str, Any]:
    axes = tuple(records[0]["baseline"]["readout"])
    primary_axes = tuple(primary_axes) if primary_axes is not None else axes
    robustness_axes = tuple(robustness_axes)

    run_rows: list[dict[str, Any]] = []
    for record in records:
        for axis in axes:
            values = [step["readout"][axis]["median"] for step in record["trajectory"]]
            run_rows.append(
                {
                    "path_id": record["path_id"],
                    "path_has_invalid_guess": record["path_has_invalid_guess"],
                    "persona_id": record["persona"]["persona_id"],
                    "quadrant": record["persona"]["quadrant"],
                    "template": record["persona"]["template"],
                    "axis": axis,
                    "values": values,
                    "slope": ols_slope(values),
                    "r5_minus_r1": values[-1] - values[0],
                }
            )

    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    invalid: dict[str, bool] = {}
    for row in run_rows:
        cells[(row["axis"], row["path_id"], row["quadrant"])].append(row["slope"])
        invalid[row["path_id"]] = row["path_has_invalid_guess"]

    path_effects: list[dict[str, Any]] = []
    for axis in axes:
        for path_id in sorted(invalid):
            quadrants = {
                quadrant: mean(cells[(axis, path_id, quadrant)])
                for quadrant in QUADRANTS
            }
            path_effects.append(
                {
                    "axis": axis,
                    "path_id": path_id,
                    "path_has_invalid_guess": invalid[path_id],
                    "quadrant_slopes": quadrants,
                    **factorial_contrasts(quadrants),
                }
            )

    inference: list[dict[str, Any]] = []
    for axis in axes:
        for effect in EFFECTS:
            values = [row[effect] for row in path_effects if row["axis"] == axis]
            valid_values = [
                row[effect]
                for row in path_effects
                if row["axis"] == axis and not row["path_has_invalid_guess"]
            ]
            inference.append(
                {
                    "axis": axis,
                    "effect": effect,
                    **descriptive(values),
                    "exact_sign_flip_p": exact_sign_flip_p(values),
                    "valid_only_sensitivity": descriptive(valid_values),
                }
            )

    for family, effects in (("overall", ("overall",)), ("moderation", MODERATORS)):
        family_rows = [
            row
            for row in inference
            if row["axis"] in primary_axes and row["effect"] in effects
        ]
        adjusted = holm_adjust(
            {
                f"{row['axis']}::{row['effect']}": row["exact_sign_flip_p"]
                for row in family_rows
            }
        )
        for row in family_rows:
            row["holm_family"] = family
            row["holm_adjusted_p"] = adjusted[f"{row['axis']}::{row['effect']}"]
    for row in inference:
        if row["axis"] in robustness_axes:
            row["holm_family"] = "robustness_descriptive"
            row["holm_adjusted_p"] = None

    trajectories = [
        {
            "axis": axis,
            "round": round_index + 1,
            **descriptive(
                [row["values"][round_index] for row in run_rows if row["axis"] == axis]
            ),
        }
        for axis in axes
        for round_index in range(5)
    ]

    return {
        "schema_version": 1,
        "record_count": len(records),
        "expected_record_count": 120,
        "complete_factorial": len(records) == 120,
        "axes": list(axes),
        "primary_axes": list(primary_axes),
        "robustness_axes": list(robustness_axes),
        "path_count": len(invalid),
        "valid_path_count": sum(not value for value in invalid.values()),
        "run_rows": run_rows,
        "path_effects": path_effects,
        "inference": inference,
        "trajectories": trajectories,
        "provenance": records[0]["provenance"],
    }


def render_summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# First-person replay summary",
        "",
        f"Records: {result['record_count']} | paths: {result['path_count']} "
        f"({result['valid_path_count']} valid-only)",
        "",
        "| concept | effect | mean slope contrast | paths +/−/0 | exact p | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["inference"]:
        holm = (
            f"{row['holm_adjusted_p']:.6f}"
            if row["holm_adjusted_p"] is not None
            else "—"
        )
        lines.append(
            f"| {row['axis']} | {row['effect']} | {row['mean']:.8f} | "
            f"{row['positive_count']}/{row['negative_count']}/{row['zero_count']} | "
            f"{row['exact_sign_flip_p']:.6f} | {holm} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--primary-axes", nargs="+", default=["happy", "sad", "angry"]
    )
    parser.add_argument("--robustness-axes", nargs="*", default=["hostile"])
    args = parser.parse_args()
    result = analyze(
        read_records(args.inputs),
        primary_axes=args.primary_axes,
        robustness_axes=args.robustness_axes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(render_summary(result), encoding="utf-8")
    print(f"analyzed {result['record_count']} records across {result['path_count']} paths")


if __name__ == "__main__":
    main()
