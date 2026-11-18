#!/usr/bin/env python3
"""Score first-person replay histories at each pre-generation boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.base_instruct_replay.replay_text import trajectory_texts
from src.probe import EmotionDirections, EmotionProbe, LocalChatModel


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--paths",
        type=Path,
        default=Path(__file__).with_name("data") / "replay_paths.json",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path(__file__).with_name("data") / "personas_first_person.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--max-personas", type=int, default=None)
    args = parser.parse_args()

    paths = load_json(args.paths)["paths"][: args.max_paths]
    personas = load_json(args.personas)[: args.max_personas]
    backend = LocalChatModel.load(args.model, device=args.device)
    directions = EmotionDirections.load(args.probe)
    probe = EmotionProbe(backend.model, backend.tokenizer, directions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for persona in personas:
            for path in paths:
                baseline_text, round_texts = trajectory_texts(persona["text"], path)
                record: dict[str, Any] = {
                    "schema_version": 1,
                    "record_kind": "base_replay_run",
                    "persona": persona,
                    "path_id": path["path_id"],
                    "path_has_invalid_guess": path["has_invalid_guess"],
                    "source_run_ids": path["source_run_ids"],
                    "baseline": {
                        "rendered_text": baseline_text,
                        "rendered_sha256": hashlib.sha256(
                            baseline_text.encode("utf-8")
                        ).hexdigest(),
                        "readout": probe.score_text(baseline_text),
                    },
                    "trajectory": [],
                    "provenance": {
                        "model": backend.metadata(),
                        "probe_path": str(args.probe.resolve()),
                        "probe_sha256": checksum(args.probe),
                        "paths_sha256": checksum(args.paths),
                        "personas_sha256": checksum(args.personas),
                        "transformers": importlib.metadata.version("transformers"),
                        "torch": importlib.metadata.version("torch"),
                    },
                }
                for round_index, rendered in enumerate(round_texts, start=1):
                    record["trajectory"].append(
                        {
                            "round": round_index,
                            "rendered_text": rendered,
                            "rendered_sha256": hashlib.sha256(
                                rendered.encode("utf-8")
                            ).hexdigest(),
                            "readout": probe.score_text(rendered),
                        }
                    )
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
                print(f"completed {persona['persona_id']} x {path['path_id']}")


if __name__ == "__main__":
    main()
