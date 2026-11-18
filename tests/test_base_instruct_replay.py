"""Tests for the Base vs. Instruct replay experiment and statistical analysis."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.base_instruct_replay.analyze_replay import analyze, read_records
from experiments.base_instruct_replay.replay_text import BOUNDARY, trajectory_texts
from src.stats import factorial_contrasts

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "base_instruct_replay"
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"


def test_personas_and_paths_schema() -> None:
    personas = json.loads((DATA_DIR / "personas_first_person.json").read_text())
    paths = json.loads((DATA_DIR / "replay_paths.json").read_text())
    assert len(personas) == 12
    assert paths["semantic_path_count"] == 10
    assert len(paths["paths"]) == 10


def test_prompt_rendering_boundaries() -> None:
    paths = json.loads((DATA_DIR / "replay_paths.json").read_text())["paths"]
    marker = "I feel worried more often."
    _, prefix_rounds = trajectory_texts(marker, paths[0])
    assert len(prefix_rounds) == 5
    assert all(text.endswith(BOUNDARY) for text in prefix_rounds)


def test_factorial_contrasts() -> None:
    effects = factorial_contrasts(
        {
            "high_e_high_n": 8.0,
            "high_e_low_n": 4.0,
            "low_e_high_n": 2.0,
            "low_e_low_n": 0.0,
        }
    )
    assert effects == {
        "overall": 3.5,
        "extraversion": 5.0,
        "neuroticism": 3.0,
        "interaction": 2.0,
    }


def test_reproduces_stored_jeong_analyses_exactly() -> None:
    for stem in (
        "base-replay-base-jeong",
        "base-replay-instruct-jeong",
    ):
        records = read_records([RESULTS_DIR / f"{stem}.jsonl"])
        recomputed = analyze(
            records,
            primary_axes=("happy", "sad", "angry"),
            robustness_axes=("hostile",),
        )
        stored = json.loads(
            (RESULTS_DIR / f"{stem}-analysis.json").read_text(encoding="utf-8")
        )
        for row_new, row_old in zip(
            recomputed["inference"], stored["inference"], strict=True
        ):
            assert row_new["axis"] == row_old["axis"]
            assert row_new["effect"] == row_old["effect"]
            assert abs(row_new["mean"] - row_old["mean"]) < 1e-12
            assert abs(
                row_new["exact_sign_flip_p"] - row_old["exact_sign_flip_p"]
            ) < 1e-12
