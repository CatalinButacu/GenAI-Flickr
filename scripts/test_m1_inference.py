"""
M1 Inference Test
=================
Tests the fine-tuned flan-T5-small scene extractor against both held-out
test-set samples and out-of-distribution free-form prompts.

Note on curly braces
--------------------
T5 SentencePiece vocabulary does not contain { or } as tokens, so they
are silently dropped during encoding/decoding.  The model correctly learns
the field names, values, and structure -- we post-process the raw output
to re-insert the missing braces before parsing.

Usage
-----
    python scripts/test_m1_inference.py
    python scripts/test_m1_inference.py --checkpoint m1_checkpoints/m1_scene_extractor
    python scripts/test_m1_inference.py --prompt "a dog chases a ball in the park"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

W = 72

# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

def load_model(checkpoint: str):
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    import torch
    path = Path(checkpoint)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"No checkpoint at '{path}'. Run train_m1_t5.py first.")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model     = T5ForConditionalGeneration.from_pretrained(str(path))
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  Loaded '{path}'  device={device}\n")
    return model, tokenizer, device


def predict(model, tokenizer, device: str, prompt: str,
            max_out: int = 512, beams: int = 4) -> str:
    import torch
    ids = tokenizer(f"extract scene: {prompt}", max_length=256,
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_length=max_out, num_beams=beams,
                             early_stopping=True, no_repeat_ngram_size=3)
    # Decode WITHOUT skipping special tokens so <extra_id_0>/<extra_id_1>
    # survive long enough for postprocess_v3 to reverse them into { / }
    raw = tokenizer.decode(out[0], skip_special_tokens=False)
    raw = postprocess_v3(raw)
    # Strip remaining T5 special tokens (pad, eos, bos, unk) but NOT extra_ids
    raw = raw.replace("</s>", "").replace("<pad>", "").replace("<s>", "").replace("<unk>", "")
    # Strip any extra_id tokens that were NOT converted (e.g. extra_id_2 and above)
    import re as _re
    raw = _re.sub(r'<extra_id_\d+>', '', raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# v3 postprocessing: reverse the <extra_id_0>/<extra_id_1> -> { / } substitution
# Used when the model was trained with train_m1_t5.py v3+ (brace-free vocab fix)
# ---------------------------------------------------------------------------

def postprocess_v3(raw: str) -> str:
    """Reverse the <extra_id_0>/<extra_id_1> substitution applied during v3 training.
    Must be called BEFORE stripping special tokens from the decoded string."""
    return raw.replace("<extra_id_0>", "{").replace("<extra_id_1>", "}")


# ---------------------------------------------------------------------------
# JSON repair:  T5 drops { and } because they are not in SentencePiece vocab
# ---------------------------------------------------------------------------

def repair_t5_json(raw: str) -> str:
    """Re-insert { } around the outer dict and each array element."""
    s = raw.strip()
    if not s:
        return s

    # Add outer braces
    if not s.startswith('{'):
        s = '{' + s
    if not s.endswith('}'):
        s += '}'

    # For each recognized section, wrap list items.
    # Items in entities start with "id", in actions with "verb",
    # in relations with "subject".
    for start_key in ('"id"', '"verb"', '"subject"'):
        esc = re.escape(start_key)
        # After '[' directly before start_key  -> '[{'
        s = re.sub(r'\[\s*' + esc, '[{' + start_key, s)
        # After ',' directly before start_key  -> '}, {'
        s = re.sub(r',\s*' + esc, '}, {' + start_key, s)

    # Close any open object just before the closing ']' of a top-level array.
    # Heuristic: a ']' that is preceded by a non-{ / non-[ character needs
    # a '}' inserted before it -- but only when we are inside a list we fixed.
    s = re.sub(r'([^{\[,\s])\s*\]\s*,\s*"(entities|actions|relations)"',
               r'\1}], "\2"', s)
    s = re.sub(r'([^{\[,\s])\s*\]\s*}$', r'\1]}', s)

    return s


def parse_json(raw: str):
    """Try v3 postprocessed, then raw, then repaired JSON. Returns (dict|None, str)."""
    v3 = postprocess_v3(raw)
    for candidate in (v3, raw, repair_t5_json(raw)):
        try:
            return json.loads(candidate), candidate
        except json.JSONDecodeError:
            pass
    return None, raw


# ---------------------------------------------------------------------------
# Fallback: extract entities by regex when JSON fails
# ---------------------------------------------------------------------------

def extract_names_regex(raw: str) -> list[str]:
    """Pull out 'name': '...' values from raw T5 output."""
    return re.findall(r'"name":\s*"([^"]+)"', raw)

def extract_relations_regex(raw: str) -> list[str]:
    """Pull out predicate values."""
    return re.findall(r'"predicate":\s*"([^"]+)"', raw)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def hr():                   print("-" * W)
def section(t):             print(f"\n  [{t}]")
def kv(k, v, i=2):         print(f"{'  '*i}{k+':':<18}{v}")
def bullet(t, c="", i=4):  print(f"{'  '*(i//2)}* {c}{t}\033[0m")

def show_result(prompt: str, raw: str) -> None:
    data, repaired = parse_json(raw)

    if data is not None:
        entities  = data.get("entities", [])
        actions   = data.get("actions", [])
        relations = data.get("relations", [])

        print(f"\n  Entities ({len(entities)}):")
        for e in entities:
            name  = e.get("name", "?")
            etype = e.get("type", "?")
            attrs = e.get("attributes", {})
            a_str = "  " + "  ".join(f"{k}={v}" for k, v in attrs.items()) if isinstance(attrs, dict) and attrs else ""
            bullet(f"\033[92m{name}\033[0m  type={etype}{a_str}")

        print(f"\n  Actions ({len(actions)}):")
        if actions:
            for a in actions:
                bullet(f"\033[94m{a.get('verb','?')}\033[0m  actor={a.get('actor','?')}")
        else:
            print("    (none)")

        print(f"\n  Relations ({len(relations)}):")
        if relations:
            for r in relations:
                bullet(f"{r.get('subject','?')}  \033[96m--[{r.get('predicate','?')}]-->\033[0m  {r.get('object','?')}")
        else:
            print("    (none)")
    else:
        # fallback: regex extraction
        names = extract_names_regex(raw)
        preds = extract_relations_regex(raw)
        print(f"\n  \033[93m[Partial parse -- JSON malformed, showing regex extraction]\033[0m")
        print(f"  Entity names  ({len(names)}):  {', '.join(names) if names else '(none found)'}")
        print(f"  Predicates   ({len(preds)}):  {', '.join(preds) if preds else '(none found)'}")
        print(f"  Raw (first 180 chars): {raw[:180]}")


def compare_counts(pred_data: dict | None, raw: str, gt: dict) -> None:
    if pred_data is None:
        pred_names = extract_names_regex(raw)
        gt_names   = [e.get("name", "") for e in gt.get("entities", [])]
        overlap    = len(set(pred_names) & set(gt_names))
        print(f"\n  Count comparison (entities only, fallback regex):")
        print(f"    pred_names = {pred_names[:6]}")
        print(f"    gt_names   = {gt_names[:6]}")
        print(f"    name overlap: {overlap}/{len(gt_names)}")
        return

    for key in ("entities", "actions", "relations"):
        pp = len(pred_data.get(key, []))
        gg = len(gt.get(key, []))
        sym = "OK" if pp == gg else "~"
        col = "\033[92m" if pp == gg else "\033[93m"
        print(f"    {col}{sym}\033[0m  {key:<12}  pred={pp}   gt={gg}")


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

FREE_FORM = [
    "a red ball falls onto a wooden table",
    "a dog chases a cat through the garden",
    "a person in a blue jacket sits on a bench next to a fountain",
    "a robot picks up a metallic box and places it on a shelf",
    "two children play with a yellow frisbee on the grass",
    "a large brown bear walks towards a river and catches a fish",
    "a woman wearing a red dress is standing near a tree",
    "a cup of coffee is on a desk next to a laptop",
]


def load_test_samples(path: str, n: int = 5) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="m1_checkpoints/m1_scene_extractor")
    ap.add_argument("--test-set",   default="data/m1_training/test.jsonl")
    ap.add_argument("--prompt",     default=None)
    ap.add_argument("--n-test",     type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()

    # Read eval_loss from checkpoint metadata if available
    meta_path = Path(args.checkpoint) / "training_metadata.json"
    eval_loss_str = "N/A"
    epochs_str = "?"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            el = meta.get("final_metrics", {}).get("eval_loss")
            ep = meta.get("epochs")
            if el is not None:
                eval_loss_str = f"{el:.4f}"
            if ep is not None:
                epochs_str = str(ep)
        except Exception:
            pass

    print("\n" + "=" * W)
    print("  M1 Scene Extractor -- Inference Test")
    print(f"  Model: flan-T5-small  fine-tuned on Visual Genome (40k training samples)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Final eval_loss: {eval_loss_str}  ({epochs_str} epochs)")
    print("=" * W + "\n")

    print("Loading model ...")
    model, tokenizer, device = load_model(args.checkpoint)

    # Single-prompt mode
    if args.prompt:
        hr()
        print(f"  PROMPT: {args.prompt}")
        hr()
        raw = predict(model, tokenizer, device, args.prompt)
        show_result(args.prompt, raw)
        hr()
        return

    # Part 1: test set
    samples = load_test_samples(args.test_set, args.n_test)
    if samples:
        print("=" * W)
        print(f"  PART 1 -- Test Set ({len(samples)} samples)  [prediction vs ground truth]")
        print("=" * W)
        for i, s in enumerate(samples, 1):
            prompt = s["input"].replace("extract scene: ", "", 1)
            gt     = json.loads(s["target"])
            hr()
            print(f"  #{i}  \"{prompt[:70]}\"")
            raw    = predict(model, tokenizer, device, prompt)
            data, _ = parse_json(raw)
            show_result(prompt, raw)
            print()
            compare_counts(data, raw, gt)

    # Part 2: free-form
    print("\n" + "=" * W)
    print("  PART 2 -- Free-form Prompts (out-of-distribution)")
    print("=" * W)
    for i, prompt in enumerate(FREE_FORM, 1):
        hr()
        print(f"  #{i}  \"{prompt}\"")
        raw = predict(model, tokenizer, device, prompt)
        show_result(prompt, raw)

    print("\n" + "=" * W)
    print("  Done.")
    print("=" * W + "\n")


if __name__ == "__main__":
    main()
