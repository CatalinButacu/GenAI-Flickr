"""
M1 Quality Benchmark
====================
Evaluates the fine-tuned flan-T5-small scene extractor on the held-out test set.

Metrics computed:
  1. JSON Syntax Rate       — % of outputs that parse as valid JSON
  2. Entity F1 (name-level) — F1 between predicted/ground-truth entity name sets
  3. Relation F1 (triple)   — F1 between predicted/ground-truth (subj, pred, obj) triples
  4. Entity Count Accuracy  — % of samples with exact entity count match
  5. Relation Count Accuracy— % of samples with exact relation count match
  6. BLEU-4 (corpus-level)  — corpus BLEU on raw JSON strings
  7. ROUGE-L (average)      — average ROUGE-L F-measure on raw JSON strings

Usage:
    python scripts/benchmark_m1_quality.py
    python scripts/benchmark_m1_quality.py --checkpoint checkpoints/scene_understanding/scene_extractor_v5
    python scripts/benchmark_m1_quality.py --n 500   # evaluate 500 samples (default: 200)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Model loading & prediction (same as test_m1_inference.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint: str):
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    import torch
    path = Path(checkpoint)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"No checkpoint at '{path}'.")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = T5ForConditionalGeneration.from_pretrained(str(path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # type: ignore[arg-type]
    model.eval()
    return model, tokenizer, device


def predict(model, tokenizer, device: str, prompt: str,
            max_out: int = 512, beams: int = 4) -> str:
    import torch
    ids = tokenizer(f"extract scene: {prompt}", max_length=256,
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_length=max_out, num_beams=beams,
                             early_stopping=True)
    raw = tokenizer.decode(out[0], skip_special_tokens=False)
    # Reverse brace substitution
    raw = raw.replace("<extra_id_0>", "{").replace("<extra_id_1>", "}")
    # Strip T5 special tokens
    raw = raw.replace("</s>", "").replace("<pad>", "").replace("<s>", "").replace("<unk>", "")
    raw = re.sub(r'<extra_id_\d+>', '', raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def try_parse(s: str) -> dict | None:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def entity_names(d: dict | None) -> set[str]:
    if d is None:
        return set()
    return {e.get("name", "").lower().strip() for e in d.get("entities", [])} - {""}


def relation_triples(d: dict | None) -> set[tuple[str, str, str]]:
    if d is None:
        return set()
    triples = set()
    for r in d.get("relations", []):
        s = r.get("subject", "").lower().strip()
        p = r.get("predicate", "").lower().strip()
        o = r.get("object", "").lower().strip()
        if s and p and o:
            triples.add((s, p, o))
    return triples


def set_f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    prec = tp / len(pred)
    rec = tp / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def _load_test_data(path: Path, n: int) -> tuple[list[dict[str, str]], int]:
    """Load JSONL test data, return (samples[:n], total_available)."""
    samples: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples[:n], len(samples)


def _run_predictions(model, tokenizer, device: str, samples, beams: int):
    """Run model inference on all samples, return raw and parsed results."""
    pred_jsons: list[str] = []
    ref_jsons: list[str] = []
    pred_dicts: list[dict | None] = []
    ref_dicts: list[dict | None] = []

    t0 = time.time()
    for i, sample in enumerate(samples):
        prompt = sample["input"].replace("extract scene: ", "", 1)
        ref_json = sample["target"]

        pred_json = predict(model, tokenizer, device, prompt, beams=beams)
        pred_jsons.append(pred_json)
        ref_jsons.append(ref_json)
        pred_dicts.append(try_parse(pred_json))
        ref_dicts.append(try_parse(ref_json))

        if (i + 1) % 50 == 0 or i == len(samples) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1:>4}/{len(samples)}]  {rate:.1f} samples/sec  "
                  f"({elapsed:.0f}s elapsed)")

    return pred_jsons, ref_jsons, pred_dicts, ref_dicts, time.time() - t0


def _compute_metrics(pred_jsons, ref_jsons, pred_dicts, ref_dicts) -> dict[str, Any]:
    """Compute all quality metrics and return as a dict."""
    n = len(pred_dicts)
    metrics: dict[str, Any] = {}

    # 1. JSON syntax rate
    n_valid = sum(1 for d in pred_dicts if d is not None)
    metrics["json_syntax_rate"] = n_valid / n

    # 2. Entity F1
    ent_f1s = [set_f1(entity_names(pd), entity_names(rd))
               for pd, rd in zip(pred_dicts, ref_dicts)]
    metrics["entity_f1"] = sum(ent_f1s) / n

    # 3. Relation F1
    rel_f1s = [set_f1(relation_triples(pd), relation_triples(rd))
               for pd, rd in zip(pred_dicts, ref_dicts)]
    metrics["relation_f1"] = sum(rel_f1s) / n

    # 4. Entity count accuracy
    metrics["entity_count_acc"] = _count_accuracy(pred_dicts, ref_dicts, "entities") / n

    # 5. Relation count accuracy
    metrics["relation_count_acc"] = _count_accuracy(pred_dicts, ref_dicts, "relations") / n

    # 6. BLEU-4
    metrics["bleu4"] = _try_bleu(ref_jsons, pred_jsons)

    # 7. ROUGE-L
    metrics["rouge_l"] = _try_rouge(ref_jsons, pred_jsons, n)

    return metrics


def _count_accuracy(preds, refs, key: str) -> int:
    """Count samples where predicted and reference list lengths match."""
    return sum(
        len((pd or {}).get(key, [])) == len((rd or {}).get(key, []))
        for pd, rd in zip(preds, refs)
    )


def _try_bleu(ref_jsons, pred_jsons) -> float | None:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs_tok = [[ref.split()] for ref in ref_jsons]
    preds_tok = [pred.split() for pred in pred_jsons]
    sf = SmoothingFunction().method1
    return float(corpus_bleu(refs_tok, preds_tok,
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=sf))


def _try_rouge(ref_jsons, pred_jsons, n: int) -> float | None:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for ref, pred in zip(ref_jsons, pred_jsons)]
    return sum(scores) / n


def _print_results(metrics: dict[str, Any], elapsed: float, n: int):
    """Pretty-print metric results table."""
    W = 72
    print()
    print("=" * W)
    print("  RESULTS")
    print("=" * W)
    print(f"  {'Metric':<30}  {'Value':>10}")
    print(f"  {'-'*30}  {'-'*10}")
    print(f"  {'JSON Syntax Rate':<30}  {metrics['json_syntax_rate']:>9.1%}")
    print(f"  {'Entity F1 (name-level)':<30}  {metrics['entity_f1']:>10.4f}")
    print(f"  {'Relation F1 (triple-level)':<30}  {metrics['relation_f1']:>10.4f}")
    print(f"  {'Entity Count Accuracy':<30}  {metrics['entity_count_acc']:>9.1%}")
    print(f"  {'Relation Count Accuracy':<30}  {metrics['relation_count_acc']:>9.1%}")
    bleu4, rouge_l = metrics["bleu4"], metrics["rouge_l"]
    if bleu4 is not None:
        print(f"  {'BLEU-4 (corpus)':<30}  {bleu4:>10.4f}")
    else:
        print(f"  {'BLEU-4 (corpus)':<30}  {'N/A':>10}  (pip install nltk)")
    if rouge_l is not None:
        print(f"  {'ROUGE-L (average)':<30}  {rouge_l:>10.4f}")
    else:
        print(f"  {'ROUGE-L (average)':<30}  {'N/A':>10}  (pip install rouge-score)")
    print(f"  {'-'*30}  {'-'*10}")
    print(f"  {'Inference time':<30}  {elapsed:>8.1f}s")
    print(f"  {'Throughput':<30}  {n/elapsed:>7.1f} s/s")
    print("=" * W)


def _save_results(metrics: dict[str, Any], elapsed: float, n: int,
                  checkpoint: str):
    """Save results to JSON file."""
    results = {
        "checkpoint": checkpoint,
        "test_samples": n,
        "json_syntax_rate": round(metrics["json_syntax_rate"], 4),
        "entity_f1": round(metrics["entity_f1"], 4),
        "relation_f1": round(metrics["relation_f1"], 4),
        "entity_count_accuracy": round(metrics["entity_count_acc"], 4),
        "relation_count_accuracy": round(metrics["relation_count_acc"], 4),
        "bleu_4": round(metrics["bleu4"], 4) if metrics["bleu4"] is not None else None,
        "rouge_l": round(metrics["rouge_l"], 4) if metrics["rouge_l"] is not None else None,
        "inference_time_s": round(elapsed, 1),
        "throughput_sps": round(n / elapsed, 1),
    }
    out_path = _ROOT / "outputs" / "m1_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path.relative_to(_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/scene_understanding/scene_extractor_v5")
    ap.add_argument("--test-file", default="data/m1_training/test.jsonl")
    ap.add_argument("--n", type=int, default=200,
                    help="Number of test samples to evaluate (default 200)")
    ap.add_argument("--beams", type=int, default=4)
    args = ap.parse_args()

    test_path = _ROOT / args.test_file
    if not test_path.exists():
        print(f"ERROR: test file not found: {test_path}")
        sys.exit(1)

    samples, total_available = _load_test_data(test_path, args.n)

    W = 72
    print("=" * W)
    print("  M1 QUALITY BENCHMARK")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Test samples: {len(samples)} / {total_available}")
    print(f"  Beam width : {args.beams}")
    print("=" * W)

    print("\nLoading model ...")
    model, tokenizer, device = load_model(args.checkpoint)
    print(f"  device={device}")

    pred_jsons, ref_jsons, pred_dicts, ref_dicts, elapsed = _run_predictions(
        model, tokenizer, device, samples, args.beams
    )

    metrics = _compute_metrics(pred_jsons, ref_jsons, pred_dicts, ref_dicts)
    _print_results(metrics, elapsed, len(samples))
    _save_results(metrics, elapsed, len(samples), args.checkpoint)


if __name__ == "__main__":
    main()
