"""
#WHERE
    Called by pipeline.py when use_t5_parser=True, and by StoryAgent/orchestrator.

#WHAT
    T5SceneParser — fine-tuned flan-T5-small (v5, eval_loss=0.0620) for
    seq2seq text → JSON scene extraction.  Falls back to PromptParser.

#INPUT
    Text prompt string.

#OUTPUT
    ParsedScene with entities, actions, spatial_relations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .prompt_parser import ParsedAction, ParsedEntity, ParsedScene, PromptParser
from src.shared.vocabulary import OBJECTS, get_object_by_keyword
from src.shared.constants import DEFAULT_SCENE_UNDERSTANDING_CHECKPOINT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour look-up (same palette as PromptParser)
# ---------------------------------------------------------------------------
_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "red":    (1.0, 0.1, 0.1, 1.0), "green":  (0.1, 0.8, 0.1, 1.0),
    "blue":   (0.1, 0.1, 1.0, 1.0), "yellow": (1.0, 0.9, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0, 1.0), "purple": (0.5, 0.0, 0.8, 1.0),
    "white":  (0.95, 0.95, 0.95, 1.0), "black": (0.1, 0.1, 0.1, 1.0),
    "gray":   (0.5, 0.5, 0.5, 1.0), "grey":   (0.5, 0.5, 0.5, 1.0),
    "brown":  (0.5, 0.3, 0.1, 1.0), "pink":   (1.0, 0.5, 0.7, 1.0),
    "cyan":   (0.0, 0.8, 0.8, 1.0),
}

_ACTOR_TYPES = {"person", "animal", "character", "human", "man", "woman",
                "child", "boy", "girl", "dog", "cat", "bird", "robot"}

_DEFAULT_MODEL = Path(DEFAULT_SCENE_UNDERSTANDING_CHECKPOINT)


class T5SceneParser:
    """Inference wrapper for the fine-tuned flan-T5-small scene extractor.

    Interface contract: ``parse(prompt) → ParsedScene`` — identical to
    ``PromptParser`` so the two are interchangeable inside ``Pipeline``.
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL,
        device: str | None = None,
        max_input_len: int = 256,
        max_output_len: int = 512,
        num_beams: int = 4,
        fallback: bool = True,
    ) -> None:
        self._path = Path(model_path)
        self._max_input = max_input_len
        self._max_output = max_output_len
        self._num_beams = num_beams
        self._fallback = fallback
        self._fallback_parser = PromptParser() if fallback else None

        # Lazy-loaded
        self._model = None
        self._tokenizer = None
        self._device: str = device or "cpu"
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load the T5 checkpoint on first use. Returns True if successful."""
        if self._is_loaded:
            return True

        config_json = self._path / "config.json"
        if not config_json.exists():
            log.warning("T5 checkpoint not found at '%s' — falling back to rules parser", self._path)
            return False

        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            if self._device == "cpu":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._path))
            self._model = T5ForConditionalGeneration.from_pretrained(str(self._path))
            self._model.to(self._device).eval()  # type: ignore[union-attr]
            self._is_loaded = True
            log.info("T5SceneParser loaded from '%s' on %s", self._path, self._device)
            return True
        except Exception as exc:
            log.warning("T5SceneParser failed to load: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API (same signature as PromptParser)
    # ------------------------------------------------------------------

    def parse(self, prompt: str) -> ParsedScene:
        """Parse *prompt* using flan-T5-small; fall back to rules on failure."""
        if not self._ensure_loaded():
            return self._do_fallback(prompt)

        try:
            raw_json = self._run_inference(prompt)
            data = self._safe_json_loads(raw_json)
            if data is None:
                log.warning("T5 output is not valid JSON — falling back")
                return self._do_fallback(prompt)
            scene = self._dict_to_parsed_scene(prompt, data)

            # Augment: if T5 found entities but 0 actions, use rules-parser
            if not scene.actions and self._fallback_parser is not None:
                fb = self._fallback_parser.parse(prompt)
                if fb.actions:
                    scene.actions = fb.actions
                    log.info("T5 found 0 actions — augmented with %d from PromptParser",
                             len(fb.actions))
                # Also merge any entities the rules-parser found that T5 missed
                t5_types = {e.object_type for e in scene.entities}
                for e in fb.entities:
                    if e.object_type not in t5_types:
                        scene.entities.append(e)
                        t5_types.add(e.object_type)
                        log.info("T5 missed entity type '%s' — added from PromptParser",
                                 e.object_type)

            return scene
        except Exception as exc:
            log.warning("T5SceneParser.parse() failed: %s — falling back", exc)
            return self._do_fallback(prompt)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, prompt: str) -> str:
        import torch

        inputs = self._tokenizer(
            f"extract scene: {prompt}",
            max_length=self._max_input,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_length=self._max_output,
                num_beams=self._num_beams,
                early_stopping=True,
                # NOTE: no_repeat_ngram_size is deliberately omitted —
                # it silently corrupts multi-entity JSON output.
            )

        # Decode WITHOUT skipping special tokens so <extra_id_0>/<extra_id_1>
        # survive long enough for _postprocess to reverse them into { / }
        raw = self._tokenizer.decode(ids[0], skip_special_tokens=False)
        raw = self._postprocess(raw)
        return raw

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess(raw: str) -> str:
        """Reverse the brace substitution and strip leftover special tokens."""
        # v3+ brace reversal
        raw = raw.replace("<extra_id_0>", "{").replace("<extra_id_1>", "}")
        # Strip standard T5 special tokens
        for tok in ("</s>", "<pad>", "<s>", "<unk>"):
            raw = raw.replace(tok, "")
        # Strip any remaining <extra_id_N> tokens
        raw = re.sub(r"<extra_id_\d+>", "", raw)
        return raw.strip()

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[dict]:
        """Attempt to parse *text* as JSON, with light repair."""
        text = text.strip()
        if not text:
            return None
        # Ensure outer braces
        if not text.startswith("{"):
            text = "{" + text
        if not text.endswith("}"):
            text += "}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Conversion:  v5 slim dict → ParsedScene
    # ------------------------------------------------------------------

    def _dict_to_parsed_scene(self, prompt: str, data: dict) -> ParsedScene:
        """Convert the v5 slim schema dict into a ``ParsedScene`` dataclass."""
        entities: List[ParsedEntity] = []
        actions: List[ParsedAction] = []
        spatial_relations: List[Dict] = []

        for e in data.get("entities", []):
            name = e.get("name", "unknown")
            etype = e.get("type", "object").lower()

            # Map raw type (e.g. "person") → OBJECTS key (e.g. "humanoid")
            if etype not in OBJECTS:
                obj_def = get_object_by_keyword(etype)
                if obj_def:
                    etype = obj_def.name
                # Also try looking up by entity name keywords
                if etype not in OBJECTS:
                    for word in name.lower().split():
                        obj_def = get_object_by_keyword(word)
                        if obj_def:
                            etype = obj_def.name
                            break

            is_actor = etype in _ACTOR_TYPES or any(
                t in name.lower() for t in _ACTOR_TYPES
            )
            # If OBJECTS recognises this as HUMANOID, force is_actor
            vocab_obj = OBJECTS.get(etype)
            if vocab_obj and vocab_obj.category.name == "HUMANOID":
                is_actor = True

            color = self._extract_color(name)
            entities.append(ParsedEntity(
                name=name,
                object_type=etype,
                is_actor=is_actor,
                color=color,
            ))

        for r in data.get("relations", []):
            subj = r.get("subject", "")
            pred = r.get("predicate", "").lower().strip()
            obj = r.get("object", "")

            # If the predicate looks like an action verb → ParsedAction
            if pred in _ACTION_PREDICATES:
                canon = _PRED_TO_ACTION.get(pred, pred)
                actions.append(ParsedAction(
                    action_type=canon,
                    actor=subj,
                    target=obj,
                ))
            else:
                spatial_relations.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                })

        return ParsedScene(
            prompt=prompt,
            entities=entities,
            actions=actions,
            spatial_relations=spatial_relations,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_color(name: str) -> Optional[Tuple[float, float, float, float]]:
        """Find a known colour word inside the entity name."""
        lower = name.lower()
        for cname, rgba in _COLORS.items():
            if cname in lower:
                return rgba
        return None

    def _do_fallback(self, prompt: str) -> ParsedScene:
        if self._fallback_parser is not None:
            return self._fallback_parser.parse(prompt)
        return ParsedScene(prompt=prompt)


# ---------------------------------------------------------------------------
# Action-like predicates that should become ParsedAction instead of spatial
# ---------------------------------------------------------------------------
_ACTION_PREDICATES = frozenset({
    "kicks", "throws", "pushes", "pulls", "hits", "catches",
    "chases", "follows", "carries", "lifts", "drops", "holds",
    "rides", "drives", "flies", "jumps", "runs", "walks",
    "sits on", "stands on", "falls on", "falls onto",
    "bounces", "rolls", "slides", "spins",
    # base forms (T5 may output either inflected or base)
    "kick", "throw", "push", "pull", "hit", "catch",
    "chase", "follow", "carry", "lift", "drop", "hold",
    "ride", "drive", "fly", "jump", "run", "walk",
    "sit", "stand", "fall", "bounce", "roll", "slide", "spin",
    "wave", "pick up", "place", "pick", "grab",
})

# Map every recognised predicate → canonical ACTIONS dict key
_PRED_TO_ACTION: dict[str, str] = {
    # inflected → base
    "kicks": "kick",    "throws": "throw",   "pushes": "push",
    "pulls": "push",    "hits": "collide",   "catches": "pick_up",
    "chases": "run",    "follows": "walk",   "carries": "pick_up",
    "lifts": "pick_up", "drops": "place",    "holds": "stand",
    "rides": "walk",    "drives": "walk",    "flies": "jump",
    "jumps": "jump",    "runs": "run",       "walks": "walk",
    "sits on": "stand", "stands on": "stand",
    "falls on": "fall", "falls onto": "fall",
    "bounces": "bounce", "rolls": "roll",    "slides": "slide",
    "spins": "walk",
    # base forms (identity mapping)
    "kick": "kick",     "throw": "throw",    "push": "push",
    "pull": "push",     "hit": "collide",    "catch": "pick_up",
    "chase": "run",     "follow": "walk",    "carry": "pick_up",
    "lift": "pick_up",  "drop": "place",     "hold": "stand",
    "ride": "walk",     "drive": "walk",     "fly": "jump",
    "jump": "jump",     "run": "run",        "walk": "walk",
    "sit": "stand",     "stand": "stand",    "fall": "fall",
    "bounce": "bounce", "roll": "roll",      "slide": "slide",
    "spin": "walk",     "wave": "wave",
    "pick up": "pick_up", "pick": "pick_up", "place": "place",
    "grab": "pick_up",
}
