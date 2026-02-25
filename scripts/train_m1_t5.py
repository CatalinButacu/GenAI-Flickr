"""Fine-tune flan-t5-small for M1 scene extraction (seq2seq: sentence → JSON).

Usage: python scripts/train_m1_t5.py [--model ...] [--epochs 15] [--max-steps 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class SceneExtractionDataset:
    """
    Loads .jsonl files produced by build_vg_dataset.py.
    Each line: {"input": "extract scene: ...", "target": "{...json...}"}
    """

    def __init__(self, jsonl_path: Path, tokenizer, max_input: int = 256, max_target: int = 512):
        self._path        = jsonl_path
        self._tokenizer   = tokenizer
        self._max_input   = max_input
        self._max_target  = max_target
        self._records: List[dict] = []

    def load(self) -> "SceneExtractionDataset":
        if not self._path.exists():
            log.error("Dataset file not found: %s", self._path)
            sys.exit(1)
        with open(self._path, encoding="utf-8") as f:
            self._records = [json.loads(line) for line in f if line.strip()]
        log.info("Loaded %d samples from %s", len(self._records), self._path)
        return self

    # ---- HuggingFace Dataset API ----

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _sub_braces(s: str) -> str:
        """Replace { } with extra_id tokens T5 already knows how to generate."""
        return s.replace("{", "<extra_id_0>").replace("}", "<extra_id_1>")

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]
        # No padding here — DataCollatorForSeq2Seq handles dynamic padding per batch
        model_inputs = self._tokenizer(
            rec["input"],
            max_length=self._max_input,
            truncation=True,
        )
        labels = self._tokenizer(
            self._sub_braces(rec["target"]),  # { -> <extra_id_0>, } -> <extra_id_1>
            max_length=self._max_target,
            truncation=True,
        )
        # Replace pad token id with -100 so loss ignores padding positions
        label_ids = [
            token_id if token_id != self._tokenizer.pad_token_id else -100
            for token_id in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs


class SceneExtractionMetrics:
    """BLEU-4, ROUGE-L, Entity F1, JSON syntax rate."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def compute(self, eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred

        # Decode tokens → strings
        predictions = self._tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [
            [t for t in label if t != -100]
            for label in labels
        ]
        label_strs = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics: Dict[str, float] = {}

        # JSON syntax rate
        valid_json = sum(1 for p in predictions if self._is_valid_json(p))
        metrics["json_syntax_rate"] = valid_json / max(len(predictions), 1)

        # ROUGE-L
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_scores = [
                scorer.score(ref, pred)["rougeL"].fmeasure
                for ref, pred in zip(label_strs, predictions)
            ]
            metrics["rouge_l"] = sum(rouge_scores) / max(len(rouge_scores), 1)
        except ImportError:
            log.warning("rouge_score not installed — skipping ROUGE-L")

        # BLEU-4
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            refs_tok  = [[ref.split()] for ref in label_strs]
            preds_tok = [pred.split() for pred in predictions]
            sf = SmoothingFunction().method1
            metrics["bleu_4"] = corpus_bleu(refs_tok, preds_tok,
                                             weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=sf)
        except ImportError:
            log.warning("nltk not installed — skipping BLEU-4")

        # Entity F1 (compares extracted entity names between pred and ref JSON)
        f1s = []
        for pred, ref in zip(predictions, label_strs):
            pred_ents = self._extract_entity_names(pred)
            ref_ents  = self._extract_entity_names(ref)
            f1s.append(self._set_f1(pred_ents, ref_ents))
        metrics["entity_f1"] = sum(f1s) / max(len(f1s), 1)

        return metrics

    # -- helpers --

    @staticmethod
    def _is_valid_json(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def _extract_entity_names(json_str: str) -> set:
        try:
            d = json.loads(json_str)
            return {e.get("name", "").lower() for e in d.get("entities", [])}
        except Exception:
            return set()

    @staticmethod
    def _set_f1(pred: set, ref: set) -> float:
        if not pred and not ref:
            return 1.0
        if not pred or not ref:
            return 0.0
        tp = len(pred & ref)
        precision = tp / len(pred)
        recall    = tp / len(ref)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class CheckpointManager:
    """Manages saving and metadata for model checkpoints."""

    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir

    @property
    def path(self) -> Path:
        return self._dir

    def save_metadata(self, metadata: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._dir / "training_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info("Saved training metadata → %s", meta_path)


class T5Trainer:
    """Orchestrates flan-t5-small fine-tuning via HuggingFace Trainer."""

    def __init__(
        self,
        model_name: str,
        data_dir: Path,
        output_dir: Path,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        max_steps: int,
        max_input_len: int,
        max_target_len: int,
        resume_from: Optional[str] = None,
    ) -> None:
        self._model_name     = model_name
        self._data_dir       = data_dir
        self._ckpt_manager   = CheckpointManager(output_dir)
        self._num_epochs     = num_epochs
        self._batch_size     = batch_size
        self._lr             = learning_rate
        self._max_steps      = max_steps
        self._max_input_len  = max_input_len
        self._max_target_len = max_target_len
        self._resume_from    = resume_from

    def run(self) -> None:
        self._check_dependencies()

        import torch
        from transformers import (
            T5ForConditionalGeneration,
            AutoTokenizer,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
        )

        log.info("Loading tokenizer + model: %s", self._model_name)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        # T5 SentencePiece vocab has no { or } tokens -- they are silently
        # dropped during encoding/decoding, breaking JSON output.
        # Fix: replace { -> <extra_id_0>  and  } -> <extra_id_1> in all targets.
        # These are already proper tokens in T5's pretrained vocabulary.
        # SceneExtractionDataset.preprocess_target() applies this substitution.
        # At inference time, reverse the substitution (extra_id_0 -> {, etc.).
        model = T5ForConditionalGeneration.from_pretrained(self._model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Training device: %s", device)
        # DO NOT call model.to(device) here! HuggingFace Trainer does it internally.
        # Moving it early causes DataCollator to generate decoder tensors on CUDA in DataLoader CPU threads, crashing Windows silently.

        # Load datasets
        train_ds = SceneExtractionDataset(
            self._data_dir / "train.jsonl", tokenizer,
            self._max_input_len, self._max_target_len,
        ).load()
        val_ds = SceneExtractionDataset(
            self._data_dir / "val.jsonl", tokenizer,
            self._max_input_len, self._max_target_len,
        ).load()

        metrics_computer = SceneExtractionMetrics(tokenizer)
        collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        # Training arguments
        out_path = str(self._ckpt_manager.path)
        training_args = Seq2SeqTrainingArguments(
            output_dir=out_path,
            num_train_epochs=self._num_epochs,
            per_device_train_batch_size=self._batch_size,
            per_device_eval_batch_size=8,              # larger eval batch for speed, but constrained to avoid OOM
            gradient_accumulation_steps=4,
            learning_rate=self._lr,
            warmup_ratio=0.05,
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="epoch",
            predict_with_generate=False,                # CRITICAL FIX: True causes silent CUDNN crashes on RTX Windows systems with T5
            generation_max_length=self._max_target_len,
            generation_num_beams=1,                     # greedy during training eval (fast)
            fp16=False,                                 # causes NaN with T5 often natively in torch
            bf16=False,                                 # RTX 3050 does not fully support bfloat16, causing silent crashes
            logging_steps=50,
            report_to="none",
            save_total_limit=2,
            dataloader_num_workers=0,                   # MUST BE 0 on Windows T5 Trainer to prevent multiprocessing hangs
            dataloader_pin_memory=False,                # CRITICAL: True causes silent Windows OOM driver crash with T5
            max_steps=self._max_steps if self._max_steps > 0 else -1,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=tokenizer,
        )

        log.info("Starting training …")
        if self._resume_from:
            log.info("Resuming from checkpoint: %s", self._resume_from)
        trainer.train(resume_from_checkpoint=self._resume_from or None)

        log.info("Saving best model to %s", out_path)
        trainer.save_model(out_path)
        tokenizer.save_pretrained(out_path)

        # Save metadata
        final_metrics = trainer.evaluate()
        self._ckpt_manager.save_metadata({
            "base_model":   self._model_name,
            "epochs":       self._num_epochs,
            "batch_size":   self._batch_size,
            "lr":           self._lr,
            "final_metrics": final_metrics,
        })
        log.info("Training complete. Final metrics: %s", final_metrics)

    @staticmethod
    def _check_dependencies() -> None:
        missing = []
        for pkg in ["transformers", "torch", "datasets"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            log.error("Missing packages: %s\nRun: pip install %s", missing, " ".join(missing))
            sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune flan-t5-small for M1 scene extraction")
    p.add_argument("--model",       default="google/flan-t5-small",        help="Base model name on HuggingFace Hub")
    p.add_argument("--data-dir",    default="data/m1_training",            help="Directory containing train.jsonl / val.jsonl")
    p.add_argument("--output",      default="m1_checkpoints/m1_scene_extractor", help="Output directory for model checkpoint")
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--max-steps",   type=int, default=0,                   help="If > 0, overrides epochs (useful for smoke tests, e.g. --max-steps 10)")
    p.add_argument("--max-input",     type=int, default=256)
    p.add_argument("--max-target",    type=int, default=256)
    p.add_argument("--resume-from",   default=None,
                   help="Path to a HuggingFace checkpoint dir to resume from (e.g. m1_checkpoints/m1_scene_extractor/checkpoint-15)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trainer = T5Trainer(
        model_name=args.model,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        max_input_len=args.max_input,
        max_target_len=args.max_target,
        resume_from=args.resume_from,
    )
    trainer.run()
