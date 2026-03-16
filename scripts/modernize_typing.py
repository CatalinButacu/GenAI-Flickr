#!/usr/bin/env python3
"""Modernize Python typing imports for Python 3.12+.

Replaces Dict→dict, List→list, Tuple→tuple, Set→set, Optional[X]→X|None, Union[X,Y]→X|Y
Adds `from __future__ import annotations` where missing.
Removes or trims `from typing import ...` lines as appropriate.
"""
from __future__ import annotations

import re
from pathlib import Path

BASE = Path(r"d:\Facultate\dissertation")

# Names that get replaced with builtins / syntax
MODERNIZE_NAMES = {"Dict", "List", "Tuple", "Optional", "Set", "Union"}

# (relative path, set of typing names to KEEP)
FILES = [
    ("src/pipeline.py", {"Any"}),
    ("src/shared/mem_profile.py", {"Generator"}),
    ("src/shared/vocabulary.py", set()),
    # understanding
    ("src/modules/understanding/t5_parser.py", set()),
    ("src/modules/understanding/retriever.py", {"Any"}),
    ("src/modules/understanding/prompt_parser.py", set()),
    # planner
    ("src/modules/planner/planner.py", {"Any"}),
    ("src/modules/planner/models.py", set()),
    ("src/modules/planner/constraint_layout.py", set()),
    # motion
    ("src/modules/motion/tokenizer.py", set()),
    ("src/modules/motion/ssm_model.py", set()),
    ("src/modules/motion/ssm_generator.py", set()),
    ("src/modules/motion/ssm/composites.py", set()),
    ("src/modules/motion/physics_trainer.py", set()),
    ("src/modules/motion/physics_dataset.py", set()),
    ("src/modules/motion/nn_models.py", set()),
    ("src/modules/motion/generator.py", set()),
    ("src/modules/motion/trainer.py", set()),
    # physics
    ("src/modules/physics/simulator.py", set()),
    ("src/modules/physics/scene.py", set()),
    ("src/modules/physics/physics_verifier.py", set()),
    ("src/modules/physics/physics_renderer.py", set()),
    ("src/modules/physics/motion_retarget.py", set()),
    ("src/modules/physics/humanoid.py", set()),
    ("src/modules/physics/camera.py", set()),
    # render
    ("src/modules/render/engine.py", set()),
    # diffusion
    ("src/modules/diffusion/controlnet_human.py", set()),
    ("src/modules/diffusion/animatediff_human.py", set()),
    # assets
    ("src/modules/assets/models.py", {"Protocol"}),
    ("src/modules/assets/generator.py", set()),
    ("src/modules/assets/backends.py", set()),
]


# ---------------------------------------------------------------------------
# Bracket-aware helpers
# ---------------------------------------------------------------------------

def find_matching_bracket(s: str, start: int) -> int:
    """Return index of ']' matching the '[' at *start*, or -1."""
    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def split_top_level(s: str, sep: str) -> list[str]:
    """Split *s* on *sep* only at bracket depth 0."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for c in s:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(c)
    parts.append("".join(current))
    return parts


# ---------------------------------------------------------------------------
# Annotation modernisation
# ---------------------------------------------------------------------------

def replace_optional(text: str) -> str:
    """Replace every ``Optional[X]`` with ``X | None``."""
    while True:
        m = re.search(r"\bOptional\[", text)
        if not m:
            break
        start = m.start()
        b_start = m.end() - 1  # the '['
        b_end = find_matching_bracket(text, b_start)
        if b_end == -1:
            break
        inner = text[b_start + 1 : b_end]
        text = text[:start] + inner + " | None" + text[b_end + 1 :]
    return text


def replace_union(text: str) -> str:
    """Replace every ``Union[X, Y, ...]`` with ``X | Y | ...``."""
    while True:
        m = re.search(r"\bUnion\[", text)
        if not m:
            break
        start = m.start()
        b_start = m.end() - 1
        b_end = find_matching_bracket(text, b_start)
        if b_end == -1:
            break
        inner = text[b_start + 1 : b_end]
        parts = split_top_level(inner, ",")
        replacement = " | ".join(p.strip() for p in parts)
        text = text[:start] + replacement + text[b_end + 1 :]
    return text


def modernize_annotations(text: str) -> str:
    """Replace Dict→dict, List→list, Tuple→tuple, Set→set, Optional, Union."""
    text = re.sub(r"\bDict\b", "dict", text)
    text = re.sub(r"\bList\b", "list", text)
    text = re.sub(r"\bTuple\b", "tuple", text)
    text = re.sub(r"\bSet\b", "set", text)
    text = replace_optional(text)
    text = replace_union(text)
    return text


# ---------------------------------------------------------------------------
# Future annotations
# ---------------------------------------------------------------------------

def ensure_future_annotations(lines: list[str]) -> list[str]:
    """Insert ``from __future__ import annotations`` if absent."""
    if any("from __future__ import annotations" in ln for ln in lines):
        return lines

    idx = 0
    # Skip shebang
    if lines and lines[0].startswith("#!"):
        idx = 1
    # Skip leading blank lines
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    # Skip module docstring
    if idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 5:
                idx += 1  # single-line docstring
            else:
                idx += 1
                while idx < len(lines):
                    if quote in lines[idx]:
                        idx += 1
                        break
                    idx += 1
    # Skip blank lines after docstring
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    lines.insert(idx, "from __future__ import annotations\n")
    # Add separator blank line if next line is non-blank
    if idx + 1 < len(lines) and lines[idx + 1].strip() != "":
        lines.insert(idx + 1, "\n")
    return lines


# ---------------------------------------------------------------------------
# Typing-import processing
# ---------------------------------------------------------------------------

def process_typing_import(line: str, keep: set[str]) -> str | None:
    """Return the modified import line, or None to delete it."""
    stripped = line.strip()
    if not stripped.startswith("from typing import"):
        return line
    imports_part = stripped[len("from typing import "):]
    names = [n.strip() for n in imports_part.split(",")]
    remaining = [n for n in names if n not in MODERNIZE_NAMES or n in keep]
    if not remaining:
        return None
    return f"from typing import {', '.join(remaining)}\n"


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------

def process_file(filepath: Path, keep: set[str]) -> bool:
    """Process one file.  Returns True if the file was changed."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    lines = content.splitlines(keepends=True)

    # 1. Ensure from __future__ import annotations
    lines = ensure_future_annotations(lines)

    # 2. Process the typing import line
    new_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("from typing import"):
            result = process_typing_import(line, keep)
            if result is not None:
                new_lines.append(result)
            # else: line removed
        else:
            new_lines.append(line)

    content = "".join(new_lines)

    # 3. Modernise type annotations in the body
    content = modernize_annotations(content)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    modified: list[str] = []
    errors: list[str] = []

    for rel_path, keep in FILES:
        fp = BASE / rel_path
        if not fp.exists():
            errors.append(f"NOT FOUND: {rel_path}")
            print(f"  NOT FOUND: {rel_path}")
            continue
        try:
            changed = process_file(fp, keep)
            status = "MODIFIED" if changed else "UNCHANGED"
            print(f"  {status}: {rel_path}")
            if changed:
                modified.append(rel_path)
        except Exception as exc:
            errors.append(f"{rel_path}: {exc}")
            print(f"  ERROR: {rel_path}: {exc}")

    print(f"\n{'=' * 60}")
    print(f"Modified {len(modified)}/{len(FILES)} files")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("No errors.")


if __name__ == "__main__":
    main()
