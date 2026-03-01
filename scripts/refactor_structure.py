#!/usr/bin/env python
"""
One-shot structural refactor:
  1. Rename src/modules/mN_* folders to clean names
  2. Move checkpoints/scene_understanding/  →  checkpoints/scene_understanding/
  3. Rename checkpoint sub-folders (strip m1_ prefix)
  4. Bulk-replace all import strings in every .py / .yaml / .bat / .md file
  5. Update constants.py checkpoint path

Run from project root:
    python scripts/refactor_structure.py [--dry-run]
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRY = False  # overridden by --dry-run

# ── 1. Module folder renames ─────────────────────────────────────────────────
MODULE_RENAMES = [
    ("scene_understanding", "scene_understanding"),
    ("scene_planner",       "scene_planner"),
    ("asset_generator",     "asset_generator"),
    ("motion_generator",    "motion_generator"),
    ("physics_engine",      "physics_engine"),
    ("render_engine",       "render_engine"),
    ("ai_enhancer",         "ai_enhancer"),
]

# ── 2. String replacements (applied to every text file) ─────────────────────
# Order matters — longest / most specific first
STRING_REPLACEMENTS = [
    # import paths
    ("src.modules.scene_understanding", "src.modules.scene_understanding"),
    ("src.modules.scene_planner",       "src.modules.scene_planner"),
    ("src.modules.asset_generator",     "src.modules.asset_generator"),
    ("src.modules.motion_generator",    "src.modules.motion_generator"),
    ("src.modules.physics_engine",      "src.modules.physics_engine"),
    ("src.modules.render_engine",       "src.modules.render_engine"),
    ("src.modules.ai_enhancer",         "src.modules.ai_enhancer"),
    # checkpoint paths (old location)
    ("checkpoints/scene_understanding/scene_extractor_v5", "checkpoints/scene_understanding/scene_extractor_v5"),
    ("checkpoints/scene_understanding/scene_extractor_v4", "checkpoints/scene_understanding/scene_extractor_v4"),
    ("checkpoints/scene_understanding/scene_extractor_v3", "checkpoints/scene_understanding/scene_extractor_v3"),
    ("checkpoints/scene_understanding/scene_extractor_v2", "checkpoints/scene_understanding/scene_extractor_v2"),
    ("checkpoints/scene_understanding/scene_extractor",    "checkpoints/scene_understanding/scene_extractor"),
    ("checkpoints/scene_understanding/",                      "checkpoints/scene_understanding/"),
    ("checkpoints/scene_understanding",                       "checkpoints/scene_understanding"),
    # docstring / comment references inside module files
    ("src/modules/scene_understanding", "src/modules/scene_understanding"),
    ("src/modules/scene_planner",       "src/modules/scene_planner"),
    ("src/modules/asset_generator",     "src/modules/asset_generator"),
    ("src/modules/motion_generator",    "src/modules/motion_generator"),
    ("src/modules/physics_engine",      "src/modules/physics_engine"),
    ("src/modules/render_engine",       "src/modules/render_engine"),
    ("src/modules/ai_enhancer",         "src/modules/ai_enhancer"),
    # leftover bare module name references (e.g. `import src.modules.motion_generator as _m4`)
    ("scene_understanding", "scene_understanding"),
    ("scene_planner",       "scene_planner"),
    ("asset_generator",     "asset_generator"),
    ("motion_generator",    "motion_generator"),
    ("physics_engine",      "physics_engine"),
    ("render_engine",       "render_engine"),
    ("ai_enhancer",         "ai_enhancer"),
]

# ── 3. Checkpoint sub-folder renames inside checkpoints/scene_understanding/ ─
CHECKPOINT_SUBDIR_RENAMES = [
    ("m1_scene_extractor_v5", "scene_extractor_v5"),
    ("m1_scene_extractor_v4", "scene_extractor_v4"),
    ("m1_scene_extractor_v3", "scene_extractor_v3"),
    ("m1_scene_extractor_v2", "scene_extractor_v2"),
    ("m1_scene_extractor",    "scene_extractor"),
]

# ── helpers ──────────────────────────────────────────────────────────────────
IGNORED_DIRS  = {".git", "__pycache__", ".mypy_cache", "wandb", "node_modules"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".bat", ".sh", ".cfg", ".ini", ".toml", ".rst"}


def apply_string_replacements(text: str) -> tuple[str, list[str]]:
    """Return (new_text, list_of_changes)."""
    changes = []
    for old, new in STRING_REPLACEMENTS:
        if old in text:
            count = text.count(old)
            text = text.replace(old, new)
            changes.append(f"  '{old}' → '{new}'  ({count}×)")
    return text, changes


def patch_file(path: Path) -> None:
    if path.suffix not in TEXT_SUFFIXES:
        return
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return
    new_text, changes = apply_string_replacements(original)
    if changes:
        print(f"  patching {path.relative_to(ROOT)}")
        for c in changes:
            print(c)
        if not DRY:
            path.write_text(new_text, encoding="utf-8")


def walk_and_patch(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for fname in filenames:
            patch_file(Path(dirpath) / fname)


def move(src: Path, dst: Path) -> None:
    if src == dst:
        return
    print(f"  MOVE  {src.relative_to(ROOT)}  →  {dst.relative_to(ROOT)}")
    if not DRY:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    global DRY
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    DRY = args.dry_run
    if DRY:
        print("=== DRY RUN — no files will be changed ===\n")

    modules_dir = ROOT / "src" / "modules"

    # ── Step 1: Patch all text files FIRST (while old names still exist) ──────
    print("── Step 1: Patching import strings in all text files ──")
    walk_and_patch(ROOT / "src")
    walk_and_patch(ROOT / "scripts")
    walk_and_patch(ROOT / "tests")
    walk_and_patch(ROOT / "examples")
    walk_and_patch(ROOT / "config")
    # root-level files
    for f in ROOT.glob("*.py"):
        patch_file(f)
    for f in ROOT.glob("*.bat"):
        patch_file(f)
    for f in ROOT.glob("*.md"):
        patch_file(f)

    # ── Step 2: Rename module folders ─────────────────────────────────────────
    print("\n── Step 2: Renaming src/modules/ folders ──")
    for old_name, new_name in MODULE_RENAMES:
        old_path = modules_dir / old_name
        new_path = modules_dir / new_name
        if old_path.exists() and not new_path.exists():
            move(old_path, new_path)
        elif new_path.exists():
            print(f"  [skip] {new_name}/ already exists")
        else:
            print(f"  [skip] {old_name}/ not found")

    # ── Step 3: Move checkpoints/scene_understanding → checkpoints/scene_understanding/ ────────
    print("\n── Step 3: Merging checkpoints/scene_understanding → checkpoints/scene_understanding ──")
    old_ckpt_root = ROOT / "checkpoints/scene_understanding"
    new_ckpt_root = ROOT / "checkpoints" / "scene_understanding"

    if old_ckpt_root.exists():
        if not DRY:
            new_ckpt_root.mkdir(parents=True, exist_ok=True)
        # Move each sub-folder, renaming to strip m1_ prefix
        for child in sorted(old_ckpt_root.iterdir()):
            new_name = child.name
            for old_sub, new_sub in CHECKPOINT_SUBDIR_RENAMES:
                if child.name == old_sub:
                    new_name = new_sub
                    break
            dst = new_ckpt_root / new_name
            if dst.exists():
                print(f"  [skip] {dst.relative_to(ROOT)} already exists")
            else:
                move(child, dst)
        # Remove now-empty checkpoints/scene_understanding/
        if not DRY and old_ckpt_root.exists():
            try:
                old_ckpt_root.rmdir()
                print(f"  rmdir  checkpoints/scene_understanding/")
            except OSError:
                print(f"  [warn] checkpoints/scene_understanding/ not empty, left in place")
    else:
        print("  checkpoints/scene_understanding/ not found — nothing to move")

    print("\nDone." + (" (DRY RUN)" if DRY else ""))


if __name__ == "__main__":
    main()
