from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .models import (
    EntityType, ExtractionResult, ExtractedEntity,
    ExtractedAction, ExtractedRelation,
)

log = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = Path("checkpoints/scene_extractor")
_TYPE_MAP = {t.name.lower(): t for t in EntityType}


class Extractor:
    """
    Fine-tuned flan-T5 extractor: text prompt → ExtractionResult (JSON seq2seq).

    Workflow:
        extractor = Extractor()
        extractor.load()            # loads weights from checkpoint
        result = extractor.extract("a red ball falls on the table")
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        max_input_len: int = 256,
        max_output_len: int = 512,
        num_beams: int = 4,
    ) -> None:
        self._path       = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT
        self._device     = device
        self._max_input  = max_input_len
        self._max_output = max_output_len
        self._num_beams  = num_beams
        self._model      = None
        self._tokenizer  = None
        self._is_loaded  = False

    # -- observable state --

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def checkpoint_path(self) -> Path:
        return self._path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str) -> None:
        self._path      = Path(value)
        self._is_loaded = False   # invalidate — must reload

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        if value not in ("cpu", "cuda", "mps"):
            raise ValueError(f"Unsupported device: {value!r}")
        self._device = value
        if self._model is not None:
            self._model.to(self._device)

    # -- public API --

    def load(self) -> None:
        """Load the fine-tuned T5 checkpoint. Raises FileNotFoundError if missing."""
        config_json = self._path / "config.json"
        if not config_json.exists():
            raise FileNotFoundError(
                f"T5 checkpoint not found at '{self._path}' (missing config.json). "
                "Run scripts/train_m1_t5.py first."
            )
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._path))
        self._model     = T5ForConditionalGeneration.from_pretrained(str(self._path))
        self._model.to(self._device).eval()
        self._is_loaded = True
        log.info("Extractor: loaded from '%s' on %s", self._path, self._device)

    def extract(self, prompt: str) -> ExtractionResult:
        """Run T5 inference and parse JSON output → ExtractionResult."""
        if not self._is_loaded:
            raise RuntimeError("Call load() before extract().")
        raw = self._run_inference(prompt)
        return self._parse_output(prompt, raw)

    # -- private --

    def _run_inference(self, prompt: str) -> str:
        import torch
        inputs = self._tokenizer(
            f"extract scene: {prompt}",
            max_length=self._max_input, truncation=True, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs, max_length=self._max_output,
                num_beams=self._num_beams, early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.decode(ids[0], skip_special_tokens=True)

    def _parse_output(self, prompt: str, raw: str) -> ExtractionResult:
        result = ExtractionResult(raw_prompt=prompt)
        data = json.loads(raw)   # raises json.JSONDecodeError if T5 output is malformed

        for e in data.get("entities", []):
            entity = ExtractedEntity(
                id=e.get("id", f"entity_{len(result.entities)}"),
                entity_type=_TYPE_MAP.get(e.get("type", "object").lower(), EntityType.OBJECT),
                name=e.get("name", ""),
                raw_span=e.get("raw", ""),
                count=e.get("count", 1),
            )
            for k, v in e.get("attributes", {}).items():
                entity.set_attr(k, v, source="ml")
            result.entities.append(entity)

        for a in data.get("actions", []):
            result.actions.append(ExtractedAction(
                verb=a.get("verb", ""),
                raw_span=a.get("raw", ""),
                actor_id=a.get("actor"),
                target_id=a.get("target"),
                parameters=a.get("parameters", {}),
            ))

        for r in data.get("relations", []):
            result.relations.append(ExtractedRelation(
                source_id=r.get("subject", ""),
                relation=r.get("predicate", ""),
                target_id=r.get("object", ""),
            ))

        return result
