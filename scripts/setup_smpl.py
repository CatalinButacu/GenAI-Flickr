#!/usr/bin/env python
"""SMPL model setup — verify installation and provide download instructions.

Run this script to check whether SMPL model files are correctly installed
and ready for use with the silhouette renderer.

Usage::

    python scripts/setup_smpl.py          # Check status
    python scripts/setup_smpl.py --help   # Show instructions
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Search directories (same as smpl_body.py)
SEARCH_DIRS = [
    ROOT / "checkpoints",
    ROOT / "models" / "smpl",
    ROOT / "models",
]

REQUIRED_FILES = {
    "SMPL_NEUTRAL": "Gender-neutral model (required)",
}
OPTIONAL_FILES = {
    "SMPL_MALE": "Male body model (optional, for gender-specific rendering)",
    "SMPL_FEMALE": "Female body model (optional, for gender-specific rendering)",
}


def _find_file(stem: str) -> Path | None:
    """Search for a model file by stem across search dirs."""
    for d in SEARCH_DIRS:
        for ext in (".npz", ".pkl"):
            path = d / f"{stem}{ext}"
            if path.is_file():
                return path
    return None


def check_numpy() -> bool:
    """Check if NumPy is available (the only required dependency)."""
    try:
        import numpy as np
        print(f"  [OK] NumPy version {np.__version__}")
        return True
    except ImportError:
        print("  [MISSING] NumPy not installed")
        print("           Fix: pip install numpy")
        return False


def check_model_files() -> tuple[bool, bool]:
    """Check for SMPL model files.  Returns (required_ok, optional_ok)."""
    print(f"  Search directories:")
    for d in SEARCH_DIRS:
        exists = "exists" if d.is_dir() else "not found"
        print(f"    {d}  ({exists})")

    req_ok = True
    print()
    for stem, desc in REQUIRED_FILES.items():
        path = _find_file(stem)
        if path is not None:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {path.name} ({size_mb:.1f} MB) — {desc}")
            print(f"       Location: {path}")
        else:
            print(f"  [MISSING] {stem}.npz — {desc}")
            req_ok = False

    opt_ok = True
    for stem, desc in OPTIONAL_FILES.items():
        path = _find_file(stem)
        if path is not None:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {path.name} ({size_mb:.1f} MB) — {desc}")
        else:
            print(f"  [---] {stem}.npz — {desc}")
            opt_ok = False

    return req_ok, opt_ok


def check_integration() -> bool:
    """Try loading the SMPL model through our wrapper."""
    try:
        from src.modules.physics.smpl_body import is_smpl_available, SMPLBody
        if not is_smpl_available():
            print("  [---] SMPL integration not ready (missing files)")
            return False
        body = SMPLBody.get_or_create(gender="neutral")
        print(f"  [OK] SMPL loaded: {body.n_vertices} vertices, {body.n_faces} faces")
        height = float(body.rest_vertices[:, 1].max() - body.rest_vertices[:, 1].min())
        print(f"  [OK] Rest-pose height: {height:.3f} m")

        # Quick forward pass check
        import numpy as np
        verts, joints = body.forward(betas=np.zeros(10))
        print(f"  [OK] Forward pass: {verts.shape[0]} vertices, {joints.shape[0]} joints")
        return True
    except Exception as e:
        print(f"  [FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_instructions():
    """Print download and setup instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    SMPL MODEL SETUP GUIDE                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Register at https://smpl.is.tue.mpg.de/                     ║
║     (Academic license — free for research use)                   ║
║                                                                  ║
║  2. Download "SMPL for Python" (version 1.1.0)                  ║
║     → You'll get a .zip with SMPL model files                   ║
║                                                                  ║
║  3. Place model files in one of these directories:               ║
║                                                                  ║
║     checkpoints/                                                 ║
║       SMPL_NEUTRAL.npz    ← required (NumPy format, ~43 MB)    ║
║       SMPL_MALE.npz       ← optional                           ║
║       SMPL_FEMALE.npz     ← optional                           ║
║                                                                  ║
║     Also accepted: models/smpl/ or models/                       ║
║     Formats: .npz (NumPy) or .pkl (pickle) — both supported    ║
║                                                                  ║
║  4. Re-run this script to verify:                                ║
║     python scripts/setup_smpl.py                                 ║
║                                                                  ║
║  Note: No additional packages are needed beyond NumPy.           ║
║  The SMPL model is licensed for academic/non-commercial use.     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

References:
  - Loper et al., "SMPL: A Skinned Multi-Person Linear Model"
    SIGGRAPH Asia 2015
  - https://github.com/DavidBoja/SMPL-Anthropometry
""")


def main():
    print("=" * 64)
    print("  SMPL Body Model — Installation Check")
    print("=" * 64)

    print("\n[1/3] NumPy (only dependency):")
    numpy_ok = check_numpy()

    print("\n[2/3] Model files:")
    req_ok, opt_ok = check_model_files()

    print("\n[3/3] Integration test:")
    if numpy_ok and req_ok:
        integ_ok = check_integration()
    else:
        print("  [SKIP] Prerequisites not met")
        integ_ok = False

    # Summary
    print("\n" + "=" * 64)
    if integ_ok:
        print("  ✓ SMPL is fully configured and ready to use!")
        print("    The silhouette renderer will use SMPL mesh bodies.")
        print("    Pure NumPy forward pass — no smplx/torch required.")
    elif numpy_ok and not req_ok:
        print("  ⚠ NumPy is available but SMPL model files are missing.")
        print("    The silhouette renderer will use the analytical fallback.")
        print_instructions()
    else:
        print("  ⚠ Dependencies not met.")
        if not numpy_ok:
            print("    → Run: pip install numpy")
        print_instructions()

    print("=" * 64)
    return 0 if integ_ok else 1


if __name__ == "__main__":
    sys.exit(main())
