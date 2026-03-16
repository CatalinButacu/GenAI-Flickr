
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

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
        checkpoint_path: str | None = None,
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
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._path))
        self._model     = T5ForConditionalGeneration.from_pretrained(str(self._path))
        self._model.to(self._device).eval()
        self._is_loaded = True
        log.info("Extractor: loaded from '%s' on %s", self._path, self._device)

    def extract(self, prompt: str) -> ExtractionResult:
        """Run T5 inference and parse JSON output → ExtractionResult."""
        if not self._is_loaded:
            raise RuntimeError("Call load() before extract().")
        raw = self.run_inference(prompt)
        return self.parse_output(prompt, raw)

    # -- private --

    def run_inference(self, prompt: str) -> str:
        inputs = self._tokenizer(
            f"extract scene: {prompt}",
            max_length=self._max_input, truncation=True, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs, max_length=self._max_output,
                num_beams=self._num_beams, early_stopping=True,
                # NOTE: no_repeat_ngram_size removed — it corrupts multi-entity JSON
            )
        # Decode with skip_special_tokens=False so <extra_id_0/1> survive
        raw = self._tokenizer.decode(ids[0], skip_special_tokens=False)
        return self.postprocess(raw)

    @staticmethod
    def postprocess(raw: str) -> str:
        """Reverse T5's brace substitution and strip special tokens.

        T5 tokenizer corrupts literal { } in JSON, so the training data
        uses <extra_id_0/1> as surrogates.  We map them back here.
        """
        raw = raw.replace("<extra_id_0>", "{").replace("<extra_id_1>", "}")
        for tok in ("</s>", "<pad>", "<s>", "<unk>"):
            raw = raw.replace(tok, "")
        raw = re.sub(r"<extra_id_\d+>", "", raw)
        return raw.strip()

    def parse_output(self, prompt: str, raw: str) -> ExtractionResult:
        result = ExtractionResult(raw_prompt=prompt)
        data = json.loads(raw)
        result.entities = self._parse_entities(data)
        self._parse_actions(result, data)
        self._parse_relations(result, data)
        return result

    def _parse_entities(self, data: dict) -> list:
        """Extract entities from parsed JSON."""
        entities = []
        for e in data.get("entities", []):
            entity = ExtractedEntity(
                id=e.get("id", f"entity_{len(entities)}"),
                entity_type=_TYPE_MAP.get(e.get("type", "object").lower(), EntityType.OBJECT),
                name=e.get("name", ""),
                raw_span=e.get("raw", e.get("name", "")),
                count=e.get("count", 1),
            )
            attrs = e.get("attributes", {})
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    entity.set_attr(k, v, source="ml")
            entities.append(entity)
        return entities

    @staticmethod
    def _parse_actions(result, data: dict) -> None:
        """Extract explicit actions from parsed JSON."""
        for a in data.get("actions", []):
            result.actions.append(ExtractedAction(
                verb=a.get("verb", ""), raw_span=a.get("raw", ""),
                actor_id=a.get("actor"), target_id=a.get("target"),
                parameters=a.get("parameters", {}),
            ))

    @staticmethod
    def _parse_relations(result, data: dict) -> None:
        """Extract relations and promote action-like predicates."""
        for r in data.get("relations", []):
            predicate = r.get("predicate", "")
            result.relations.append(ExtractedRelation(
                source_id=r.get("subject", r.get("source_id", "")),
                relation=predicate,
                target_id=r.get("object", r.get("target_id", "")),
            ))
            if predicate in _ACTION_VERBS:
                result.actions.append(ExtractedAction(
                    verb=predicate,
                    actor_id=r.get("subject", ""),
                    target_id=r.get("object", ""),
                ))


_ACTION_VERBS = frozenset({
    "kicks", "throws", "pushes", "pulls", "hits", "catches",
    "chases", "follows", "carries", "lifts", "drops", "holds",
    "rides", "drives", "flies", "jumps", "runs", "walks",
    "falls on", "falls onto", "bounces", "rolls", "slides",
})
