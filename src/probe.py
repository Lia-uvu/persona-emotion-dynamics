"""Model loading and activation-space emotion direction scoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np


def local_snapshot_checksum(path: str | Path) -> str:
    """Compute SHA-256 over model weights and tokenizer/config files."""
    source = Path(path)
    patterns = (
        "*.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    files = sorted({file for pattern in patterns for file in source.glob(pattern)})
    if not files:
        raise ValueError(f"no model snapshot files found in {source}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class LocalChatModel:
    """Wrapper around a Hugging Face causal LM for forward passes and seeded generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        device: Any,
        snapshot_checksum: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.snapshot_checksum = snapshot_checksum

    @classmethod
    def load(
        cls,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
    ) -> "LocalChatModel":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        target = torch.device(device)
        dtype = torch.float16 if target.type in {"mps", "cuda"} else torch.float32
        local_path = Path(model_name)
        snapshot_checksum = (
            local_snapshot_checksum(local_path) if local_path.is_dir() else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(target)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=target,
            snapshot_checksum=snapshot_checksum,
        )

    def metadata(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "name": self.model_name,
            "model_type": getattr(config, "model_type", None),
            "revision": getattr(config, "_commit_hash", None),
            "snapshot_sha256": self.snapshot_checksum,
            "dtype": str(getattr(self.model, "dtype", "unknown")),
            "device": str(self.device),
        }


@dataclass
class EmotionDirections:
    model_type: str
    model_checksum: str | None
    materials_checksum: str
    directions: dict[str, dict[int, np.ndarray]]
    validation: dict[str, Any]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for axis, layers in self.directions.items():
            for layer, direction in layers.items():
                arrays[f"{axis}__{layer}"] = direction.astype(np.float32)
        arrays["__metadata__"] = np.asarray(
            json.dumps(
                {
                    "model_type": self.model_type,
                    "model_checksum": self.model_checksum,
                    "materials_checksum": self.materials_checksum,
                    "validation": self.validation,
                },
                ensure_ascii=False,
            )
        )
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "EmotionDirections":
        data = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(data["__metadata__"]))
        directions: dict[str, dict[int, np.ndarray]] = {}
        for key in data.files:
            if key == "__metadata__":
                continue
            axis, layer = key.split("__", 1)
            directions.setdefault(axis, {})[int(layer)] = data[key].astype(np.float32)
        return cls(
            model_type=metadata["model_type"],
            model_checksum=metadata.get("model_checksum"),
            materials_checksum=metadata["materials_checksum"],
            directions=directions,
            validation=metadata["validation"],
        )


def cosine_projection(hidden: np.ndarray, direction: np.ndarray) -> float:
    magnitude = float(np.linalg.norm(direction) * np.linalg.norm(hidden))
    if not magnitude:
        raise ValueError("direction or hidden vector has zero magnitude")
    return float(hidden @ direction / magnitude)


class EmotionProbe:
    """Score activation-space cosine alignment with pre-extracted concept directions."""

    def __init__(self, model: Any, tokenizer: Any, directions: EmotionDirections):
        if directions.model_type != model.config.model_type:
            raise ValueError(
                f"direction model_type {directions.model_type!r} does not match "
                f"model {model.config.model_type!r}"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.directions = directions
        self.axes = tuple(
            axis for axis, layers in directions.directions.items() if layers
        )

    def _hidden_states(self, texts: Sequence[str]) -> dict[int, np.ndarray]:
        from repeng.extract import batched_get_hiddens

        layers = sorted(
            {
                layer
                for axis in self.axes
                for layer in self.directions.directions[axis]
            }
        )
        return batched_get_hiddens(
            self.model, self.tokenizer, list(texts), layers, batch_size=2
        )

    def score_text(self, rendered_text: str) -> dict[str, Any]:
        hidden = self._hidden_states([rendered_text])
        result: dict[str, Any] = {}
        for axis in self.axes:
            by_layer = {
                str(layer): cosine_projection(hidden[layer][0], direction)
                for layer, direction in self.directions.directions[axis].items()
            }
            result[axis] = {
                "layers": by_layer,
                "median": float(median(by_layer.values())),
            }
        return result
