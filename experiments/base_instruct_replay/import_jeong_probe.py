#!/usr/bin/env python3
"""Import published Jeong Neural-MRI comprehension vectors into .npz probe format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.probe import EmotionDirections, local_snapshot_checksum

SOURCE_REPOSITORY = "https://github.com/JihoonJeong/Neural-MRI"
SOURCE_COMMIT = "538b8af2063f6eded9dabaadb1a0f407bf04eb6e"
SOURCE_LAYER = 15
CONCEPTS = ("happy", "sad", "angry", "hostile")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    layer = payload["vectors"][str(SOURCE_LAYER)]
    directions = {
        concept: {SOURCE_LAYER: layer[concept].detach().cpu().float().numpy()}
        for concept in CONCEPTS
    }

    source_sha = checksum(args.source)
    provenance = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_artifact_sha256": source_sha,
        "source_metadata_sha256": checksum(args.metadata),
        "source_model": metadata["model_id"],
        "source_layer": SOURCE_LAYER,
        "source_variant": "vectors_comprehension.pt",
        "source_concepts": list(CONCEPTS),
        "published_best_layer_cosine": metadata["best_layer_cosine"],
    }
    artifact = EmotionDirections(
        model_type="qwen2",
        model_checksum=local_snapshot_checksum(args.model),
        materials_checksum=f"external:{SOURCE_COMMIT}:{source_sha}",
        directions=directions,
        validation={concept: dict(provenance) for concept in CONCEPTS},
    )
    artifact.save(args.output)
    print(f"wrote Jeong layer-{SOURCE_LAYER} probe to {args.output}")


if __name__ == "__main__":
    main()
