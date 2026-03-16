"""AMASS/ARCTIC sample retriever for pipeline use before models are trained.

Loads motion samples and returns them as MotionClips with SMPL-X pose
parameters (168-dim frames) and optionally forward-kinematics joint positions.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from src.shared.data.smplx_loader import AMASSLoader, ARCTICLoader, SMPLXSample
from src.shared.constants import DEFAULT_DATA_DIR, ARCTIC_DATA_DIR, MOTION_FPS
from .models import MotionClip

log = logging.getLogger(__name__)

# ── Action keyword groups for scoring ────────────────────────────────────────
_ACTION_HINTS: dict[str, tuple[str, ...]] = {
    "walk": ("walk", "walking", "stroll", "step", "tread", "wander"),
    "run":  ("run", "running", "jog", "jogging", "sprint", "dash"),
    "jump": ("jump", "jumping", "leap", "hop"),
    "sit":  ("sit", "sitting", "seat", "squat"),
    "stand": ("stand", "standing", "upright", "idle", "rest"),
    "wave": ("wave", "waving", "greet", "gesture"),
    "punch": ("punch", "punching", "hit", "strike"),
    "kick": ("kick", "kicking"),
    "dance": ("dance", "dancing"),
    "throw": ("throw", "throwing", "toss"),
}

# Clips whose names suggest lying flat / non-upright — penalise for walking prompts
_UPRIGHT_NEGATIVE: frozenset[str] = frozenset({
    "lie", "lying", "lay", "supine", "prone", "crawl", "crawling",
    "crouch", "crouching", "kneel", "kneeling", "fall", "falling",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase word-tokens + simple stem (strip trailing 's' from >3-char tokens)."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    stemmed = set()
    for t in tokens:
        stemmed.add(t)
        if len(t) > 3 and t.endswith("s"):
            stemmed.add(t[:-1])
        if len(t) > 4 and t.endswith("ing"):
            stemmed.add(t[:-3])
        if len(t) > 4 and t.endswith("ed"):
            stemmed.add(t[:-2])
    return stemmed


def _score_sample(sample: SMPLXSample, query_tokens: set[str]) -> float:
    """Score a sample against the tokenised query. Higher = better match."""
    # Use sample_id (e.g. 'ACCAD/Female1Walking_c3d/B7_-_walk_backwards') for scoring
    src = (sample.sample_id or "").lower()
    src_tokens = _tokenize(src)
    score = 0.0

    # Token overlap
    overlap = query_tokens & src_tokens
    score += len(overlap) * 2.0

    # Detect which action group the sample belongs to
    sample_action: str | None = None
    for action, hints in _ACTION_HINTS.items():
        if any(h in src_tokens for h in hints):
            sample_action = action
            break

    # Detect which action group the query requests
    query_action: str | None = None
    for action, hints in _ACTION_HINTS.items():
        if any(h in query_tokens for h in hints):
            query_action = action
            break

    # Reward matching action group
    if query_action and sample_action == query_action:
        score += 8.0

    # Penalise upright-negative clips when the query doesn't mention lying/crawling
    if _UPRIGHT_NEGATIVE & src_tokens and not (_UPRIGHT_NEGATIVE & query_tokens):
        score -= 6.0

    # Heavy penalty for walk/run cross-match
    walk_query = query_action == "walk"
    run_query   = query_action == "run"
    if walk_query and sample_action == "run":
        score -= 10.0
    if run_query and sample_action == "walk":
        score -= 10.0

    return score


def _select_sample(samples: list[SMPLXSample], text: str) -> SMPLXSample:
    """Return the highest-scoring sample for *text*."""
    query_tokens = _tokenize(text)
    best_sample = samples[0]
    best_score = float("-inf")
    scores = []
    for s in samples:
        sc = _score_sample(s, query_tokens)
        scores.append(sc)
        if sc > best_score:
            best_score = sc
            best_sample = s
    shortlist = sum(1 for s in scores if s == best_score)
    log.info(
        "AMASSSampleRetriever: selected %s (score=%.2f, shortlist=%d)",
        best_sample.sample_id, best_score, shortlist,
    )
    return best_sample


def _resample_motion_window(
    sample: SMPLXSample,
    out_frames: int,
) -> np.ndarray:
    """Select and resample a window of *sample.motion* to exactly *out_frames* frames
    at MOTION_FPS (regardless of the source recording fps)."""
    source_fps = float(sample.fps) if sample.fps else float(MOTION_FPS)
    total_source = sample.motion.shape[0]
    # How many source frames correspond to the desired output at MOTION_FPS
    source_window = min(total_source, max(1, int(out_frames * source_fps / MOTION_FPS)))
    source_clip = sample.motion[:source_window]
    if source_window == out_frames:
        return source_clip
    indices = np.round(np.linspace(0, source_window - 1, out_frames)).astype(int)
    return source_clip[indices]


class AMASSSampleRetriever:
    """Retrieve real AMASS/ARCTIC motion samples for pipeline testing.

    Until the motion generation model is trained, this provides real motion
    data from the dataset as sample output.  Selection is scoring-based so
    a walking prompt always returns a walking clip, etc.
    """

    def __init__(
        self,
        amass_dir: str = DEFAULT_DATA_DIR,
        arctic_dir: str = ARCTIC_DATA_DIR,
        max_samples: int = 200,
    ):
        self._samples: list[SMPLXSample] = []
        self._smplx_body = None
        self._load_samples(amass_dir, arctic_dir, max_samples)

    def _load_samples(self, amass_dir: str, arctic_dir: str, max_samples: int) -> None:
        """Load a subset of AMASS + ARCTIC for retrieval."""
        from pathlib import Path

        if Path(amass_dir).exists():
            loader = AMASSLoader(amass_dir)
            self._samples.extend(loader.load_dataset(
                max_samples=max_samples, min_frames=30,
            ))

        arctic_raw = Path(arctic_dir) / "raw_seqs"
        if arctic_raw.exists():
            arctic = ARCTICLoader(arctic_dir)
            self._samples.extend(arctic.load_dataset(
                max_samples=min(max_samples // 4, 50), min_frames=30,
            ))

        log.info("AMASSSampleRetriever: loaded %d samples", len(self._samples))

    def retrieve(self, text: str, max_frames: int = 200) -> MotionClip | None:
        """Return the best-matching AMASS/ARCTIC sample as a MotionClip.

        Uses scoring-based selection (token overlap + action penalties) to
        pick a clip that matches the text prompt.  The clip is resampled to
        *max_frames* at MOTION_FPS regardless of source recording fps.
        """
        if not self._samples:
            return None

        sample = _select_sample(self._samples, text)
        motion = _resample_motion_window(sample, max_frames)

        # Get joint positions via forward kinematics if body model available
        raw_joints = self._get_joint_positions(motion, sample.betas)

        return MotionClip(
            action=text,
            frames=motion,
            fps=MOTION_FPS,            # normalised -- always MOTION_FPS (30)
            source=f"sample_{sample.sample_id}",
            raw_joints=raw_joints,
            betas=sample.betas,
        )

    def _get_joint_positions(
        self,
        motion: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray | None:
        """Compute (T, 55, 3) joint positions via SMPL-X forward kinematics."""
        if self._smplx_body is None:
            try:
                from src.modules.physics.smplx_body import SMPLXBody
                self._smplx_body = SMPLXBody.get_or_create("neutral")
            except FileNotFoundError:
                log.warning("SMPL-X body model not found — raw_joints unavailable")
                return None

        return self._smplx_body.get_joint_positions_batch(motion, betas)
