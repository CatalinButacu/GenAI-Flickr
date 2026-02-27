"""
Constraint-based spatial layout solver for M2.

Maps M1 relation triples (subject, predicate, object) to 3D position
constraints and solves for entity positions using scipy.optimize.minimize.

Supported spatial predicates:
    on, on top of     → stack vertically (z-axis alignment)
    above, over       → subject above object with gap
    below, under      → subject below object
    in front of       → subject closer to camera (y < y_obj)
    behind            → subject further from camera (y > y_obj)
    beside, next to   → side-by-side (x offset, same z)
    near, by          → close proximity (soft)
    left of           → subject.x < object.x
    right of          → subject.x > object.x
    in, inside        → same position (containment)
    has, wearing, of  → ignored (non-spatial)

Falls back to row-based placement when no solvable constraints exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predicate → constraint type mapping
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class SpatialConstraint:
    """One constraint between two named entities."""
    subject: str
    object: str
    ctype: str  # ON, ABOVE, BELOW, FRONT, BEHIND, BESIDE, NEAR, LEFT, RIGHT, INSIDE


# Predicate aliases → canonical constraint type
_PRED_MAP: Dict[str, str] = {
    "on":           "ON",
    "on top of":    "ON",
    "atop":         "ON",
    "above":        "ABOVE",
    "over":         "ABOVE",
    "below":        "BELOW",
    "under":        "BELOW",
    "beneath":      "BELOW",
    "in front of":  "FRONT",
    "in front":     "FRONT",
    "behind":       "BEHIND",
    "beside":       "BESIDE",
    "next to":      "BESIDE",
    "near":         "NEAR",
    "by":           "NEAR",
    "across":       "NEAR",
    "left of":      "LEFT",
    "to the left":  "LEFT",
    "right of":     "RIGHT",
    "to the right": "RIGHT",
    "in":           "INSIDE",
    "inside":       "INSIDE",
    "inside of":    "INSIDE",
}

# Non-spatial predicates to skip
_SKIP_PREDS = {"has", "have", "wearing", "of", "with", "is", "are",
               "has a", "has an", "part of", "full of"}

# Constraint weights
_W = {
    "ON":     10.0,    # strong vertical stacking
    "ABOVE":   5.0,
    "BELOW":   5.0,
    "FRONT":   5.0,
    "BEHIND":  5.0,
    "BESIDE":  5.0,
    "NEAR":    3.0,    # softer
    "LEFT":    5.0,
    "RIGHT":   5.0,
    "INSIDE":  3.0,
}

# Layout constants
_STACK_GAP   = 0.25   # vertical gap for ON/ABOVE
_SIDE_OFFSET = 0.5    # horizontal offset for BESIDE/LEFT/RIGHT
_NEAR_DIST   = 0.7    # target distance for NEAR
_REPULSION   = 2.0    # overlap repulsion weight
_MIN_DIST    = 0.3    # minimum distance between entities
_GROUND_Z    = 0.15   # default ground level


# ---------------------------------------------------------------------------
# Constraint builder
# ---------------------------------------------------------------------------

def build_constraints(relations: List[Dict]) -> List[SpatialConstraint]:
    """Convert M1 relation triples to spatial constraints.

    Args:
        relations: List of {"subject": str, "predicate": str, "object": str}

    Returns:
        List of SpatialConstraint objects (non-spatial predicates filtered out).
    """
    return [c for rel in relations if (c := _parse_one_constraint(rel)) is not None]


def _parse_one_constraint(rel: Dict) -> SpatialConstraint | None:
    """Parse a single relation dict into a SpatialConstraint, or None."""
    subj = rel.get("subject", "").strip().lower()
    pred = rel.get("predicate", "").strip().lower()
    obj  = rel.get("object", "").strip().lower()
    if not subj or not pred or not obj:
        return None
    if pred in _SKIP_PREDS:
        return None
    ctype = _resolve_predicate(pred)
    if ctype is None:
        log.debug("Skipping unknown predicate: %r", pred)
        return None
    return SpatialConstraint(subject=subj, object=obj, ctype=ctype)


def _resolve_predicate(pred: str) -> str | None:
    """Map a predicate string to its canonical constraint type."""
    ctype = _PRED_MAP.get(pred)
    if ctype is not None:
        return ctype
    # Partial match for compound predicates like "standing on"
    for key, ct in _PRED_MAP.items():
        if key in pred:
            return ct
    return None


# ---------------------------------------------------------------------------
# Energy function for scipy.optimize
# ---------------------------------------------------------------------------

def _cost_on(sx, sy, sz, ox, oy, oz):
    """Subject directly on top of object: align XY, stack Z."""
    return ((sx - ox) ** 2 + (sy - oy) ** 2) + (sz - oz - _STACK_GAP) ** 2


def _cost_above(sx, sy, sz, ox, oy, oz):
    return max(0, oz + _STACK_GAP - sz) ** 2


def _cost_below(sx, sy, sz, ox, oy, oz):
    return max(0, sz - oz + _STACK_GAP) ** 2


def _cost_front(sx, sy, sz, ox, oy, oz):
    return max(0, sy - oy + _SIDE_OFFSET) ** 2


def _cost_behind(sx, sy, sz, ox, oy, oz):
    return max(0, oy - sy + _SIDE_OFFSET) ** 2


def _cost_beside(sx, sy, sz, ox, oy, oz):
    return (abs(sx - ox) - _SIDE_OFFSET) ** 2 + (sz - oz) ** 2


def _cost_near(sx, sy, sz, ox, oy, oz):
    dist = np.sqrt((sx - ox)**2 + (sy - oy)**2 + (sz - oz)**2)
    return (dist - _NEAR_DIST) ** 2


def _cost_left(sx, sy, sz, ox, oy, oz):
    return max(0, sx - ox + _SIDE_OFFSET) ** 2


def _cost_right(sx, sy, sz, ox, oy, oz):
    return max(0, ox - sx + _SIDE_OFFSET) ** 2


def _cost_inside(sx, sy, sz, ox, oy, oz):
    return (sx - ox)**2 + (sy - oy)**2 + (sz - oz)**2


# Dispatch table: constraint type → cost function
_COST_FN = {
    "ON": _cost_on, "ABOVE": _cost_above, "BELOW": _cost_below,
    "FRONT": _cost_front, "BEHIND": _cost_behind, "BESIDE": _cost_beside,
    "NEAR": _cost_near, "LEFT": _cost_left, "RIGHT": _cost_right,
    "INSIDE": _cost_inside,
}


def _repulsion_cost(pos: np.ndarray, n: int) -> float:
    """Penalty that prevents entity overlap."""
    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((pos[i] - pos[j]) ** 2)) + 1e-6
            if dist < _MIN_DIST:
                cost += _REPULSION * (_MIN_DIST / dist - 1.0) ** 2
    return cost


def _ground_cost(pos: np.ndarray, n: int) -> float:
    """Soft penalty for entities below ground level."""
    cost = 0.0
    for i in range(n):
        if pos[i, 2] < _GROUND_Z:
            cost += 2.0 * (_GROUND_Z - pos[i, 2]) ** 2
    return cost


def _build_energy(
    entity_names: List[str],
    constraints: List[SpatialConstraint],
    name_to_idx: Dict[str, int],
) -> callable:
    """Build an energy function mapping flat position vector → scalar cost."""

    n = len(entity_names)

    def energy(x: np.ndarray) -> float:
        pos = x.reshape(n, 3)
        cost = 0.0

        for c in constraints:
            si = name_to_idx.get(c.subject)
            oi = name_to_idx.get(c.object)
            if si is None or oi is None:
                continue
            fn = _COST_FN.get(c.ctype)
            if fn is None:
                continue
            w = _W.get(c.ctype, 3.0)
            cost += w * fn(*pos[si], *pos[oi])

        cost += _repulsion_cost(pos, n)
        cost += _ground_cost(pos, n)
        return cost

    return energy


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_layout(
    entity_names: List[str],
    relations: List[Dict],
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Solve for 3D positions of entities given relation constraints.

    Args:
        entity_names: List of entity name strings.
        relations: M1-style relation dicts with subject, predicate, object.
        seed: Random seed for reproducible initial positions.

    Returns:
        Dict mapping entity name → (x, y, z) position tuple.
    """
    if not entity_names:
        return {}

    constraints = build_constraints(relations)

    # If no solvable constraints, return None to signal fallback
    if not constraints:
        log.info("No spatial constraints found — falling back to row layout")
        return {}

    # Build name → index mapping (handle entities mentioned in relations
    # but not in the entity list by ignoring them)
    name_to_idx: Dict[str, int] = {}
    for i, name in enumerate(entity_names):
        name_to_idx[name.lower()] = i

    n = len(entity_names)
    rng = np.random.default_rng(seed)

    # Initialize: spread entities on a grid + small noise
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[i * 3 + 0] = (i % 3 - 1) * _SIDE_OFFSET + rng.uniform(-0.1, 0.1)  # x
        x0[i * 3 + 1] = (i // 3) * _SIDE_OFFSET + rng.uniform(-0.1, 0.1)      # y
        x0[i * 3 + 2] = _GROUND_Z + rng.uniform(0, 0.05)                        # z

    energy_fn = _build_energy(entity_names, constraints, name_to_idx)

    result = minimize(
        energy_fn, x0,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not result.success:
        log.warning("Layout solver did not converge: %s", result.message)

    positions = result.x.reshape(n, 3)
    solved: Dict[str, Tuple[float, float, float]] = {}
    for i, name in enumerate(entity_names):
        solved[name] = (
            round(float(positions[i, 0]), 3),
            round(float(positions[i, 1]), 3),
            round(float(max(positions[i, 2], 0.05)), 3),  # keep above ground
        )

    log.info("Constraint solver: %d constraints, %d entities, cost=%.4f",
             len(constraints), n, result.fun)
    return solved
