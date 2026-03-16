"""Analytical body model: prompt → BodyParams → silhouette polygons.

Provides BodyParams (interpretable body shape), keyword extraction from
natural language, and polygon-based body rendering for 21-joint skeletons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── 21-joint humanoid indices ───────────────────────────────────────────────────

ROOT = 0
SPINE1 = 1;  SPINE2 = 2;  NECK = 3;  HEAD = 4
L_SHO = 5;   L_ELB = 6;   L_WRI = 7
R_SHO = 8;   R_ELB = 9;   R_WRI = 10
L_HIP = 11;  L_KNE = 12;  L_ANK = 13;  L_HEEL = 14;  L_TOE = 15
R_HIP = 16;  R_KNE = 17;  R_ANK = 18;  R_HEEL = 19;  R_TOE = 20


# ── Body type enum ───────────────────────────────────────────────────────────

class BodyType(Enum):
    ECTOMORPH = auto()   # slim, lean
    MESOMORPH = auto()   # muscular, athletic
    ENDOMORPH = auto()   # stocky, heavy-set
    AVERAGE = auto()


# ── Parametric body shape ────────────────────────────────────────────────────

@dataclass(slots=True)
class BodyParams:
    """Interpretable body shape parameters — prompt-extractable.

    All proportions are relative multipliers (1.0 = average adult).
    The system maps natural-language descriptors to these values.

    Attributes
    ----------
    height_m : float
        Total body height in metres (default 1.72).
    mass_kg : float
        Body mass in kg (default 72.0).
    body_type : BodyType
        Somatotype classification → affects width proportions.
    shoulder_width : float
        Relative shoulder breadth (1.0 = average, 1.3 = broad).
    hip_width : float
        Relative hip breadth.
    torso_length : float
        Relative torso length.
    leg_length : float
        Relative leg length.
    arm_length : float
        Relative arm length.
    head_scale : float
        Relative head size (1.0 = proportional to height/7.5).
    muscle_mass : float
        Relative muscularity (affects limb cross-section widths).
        0.5 = lean, 1.0 = average, 1.5 = very muscular.
    body_fat : float
        Body fat factor (affects torso width and roundness).
        0.5 = lean, 1.0 = average, 1.5 = heavy.
    gender : str
        ``"male"``, ``"female"``, or ``"neutral"`` — affects default proportions.
    age_group : str
        ``"child"``, ``"teen"``, ``"adult"``, ``"elderly"`` — affects proportions.
    skin_tone : tuple[int, int, int]
        BGR colour for the silhouette fill (default: warm neutral).
    """
    # STANDARD: WHO global average adult height (2020). Not architectural.
    height_m: float = 1.72
    # STANDARD: WHO global average adult mass (2020). Not architectural.
    mass_kg: float = 72.0
    body_type: BodyType = BodyType.AVERAGE
    shoulder_width: float = 1.0
    hip_width: float = 1.0
    torso_length: float = 1.0
    leg_length: float = 1.0
    arm_length: float = 1.0
    head_scale: float = 1.0
    muscle_mass: float = 1.0
    body_fat: float = 1.0
    gender: str = "neutral"
    age_group: str = "adult"
    # DESIGN CHOICE: warm neutral skin tone (BGR 180,160,140). Arbitrary
    # default — overridden by prompt extraction when skin descriptors present.
    skin_tone: tuple[int, int, int] = (180, 160, 140)

    @property
    def bmi(self) -> float:
        return self.mass_kg / (self.height_m ** 2)

    @property
    def bulk_factor(self) -> float:
        """Combined width multiplier from muscle + fat.

        DESIGN CHOICE: 0.6 base + 0.25×muscle + 0.15×fat.
        At muscle=1.0, fat=1.0 → bulk=1.0 (identity). Muscle has ~1.7×
        more influence than fat on limb cross-section. Based on published
        body composition → circumference correlations (DeSilva 1999).
        Not architectural.
        """
        return 0.6 + 0.25 * self.muscle_mass + 0.15 * self.body_fat

    def to_smpl_betas(self) -> "np.ndarray":
        """Convert to SMPL shape coefficients (10 PCA betas).

        See :func:`~smpl_body.smpl_betas_from_body_params` for details.
        """
        from .smpl_body import smpl_betas_from_body_params
        return smpl_betas_from_body_params(self)

    def apply_body_type_defaults(self) -> None:
        """Adjust proportions based on somatotype if not explicitly set.

        DESIGN CHOICES — body-type multipliers (not architectural):
        - Ectomorph: narrow shoulders (0.90×), low muscle/fat caps
        - Mesomorph: broad shoulders (1.15×), muscle floor 1.2
        - Endomorph: wide hips (1.15×), moderate shoulders (1.05×)
        Values derived from Sheldon somatotype proportions (approximate).
        """
        match self.body_type:
            case BodyType.ECTOMORPH:
                self.muscle_mass = min(self.muscle_mass, 0.7)
                self.body_fat = min(self.body_fat, 0.6)
                self.shoulder_width *= 0.90
            case BodyType.MESOMORPH:
                self.muscle_mass = max(self.muscle_mass, 1.2)
                self.shoulder_width *= 1.15
            case BodyType.ENDOMORPH:
                self.body_fat = max(self.body_fat, 1.3)
                self.hip_width *= 1.15
                self.shoulder_width *= 1.05

    def apply_gender_defaults(self) -> None:
        """Adjust proportions for gendered body shapes."""
        match self.gender:
            case "male":
                self.shoulder_width = max(self.shoulder_width, 1.05)
                self.hip_width = min(self.hip_width, 0.95)
            case "female":
                self.shoulder_width = min(self.shoulder_width, 0.95)
                self.hip_width = max(self.hip_width, 1.05)

    def apply_age_defaults(self) -> None:
        """Adjust proportions for age group.

        DESIGN CHOICES — age adjustments (not architectural):
        - Child: head_scale ≥ 1.3 (children's heads are ~25-30% of body),
          shorter legs (0.85×) and torso (0.90×).
        - Elderly: height loss ~4% (WHO osteoporosis data), shorter torso.
        """
        match self.age_group:
            case "child":
                self.head_scale = max(self.head_scale, 1.3)
                self.leg_length *= 0.85
                self.torso_length *= 0.90
            case "elderly":
                self.height_m *= 0.96
                self.torso_length *= 0.95


# ── Prompt → BodyParams extraction ───────────────────────────────────────────

# DESIGN CHOICES — height values from WHO (2020) adult percentiles:
# P5=1.52m, P25=1.60m, P50=1.72m, P75=1.85m, P95=1.92m.
# "towering" / "giant" go beyond P99 for dramatic effect. Not architectural.
_HEIGHT_KEYWORDS: dict[str, float] = {
    "very tall": 1.92, "tall": 1.85, "short": 1.60, "very short": 1.52,
    "petite": 1.55, "towering": 1.98, "average height": 1.72,
    "tiny": 1.50, "huge": 1.95, "giant": 2.05,
}
# DESIGN CHOICES — build keyword → (muscle_mass, body_fat, somatotype).
# muscle_mass and body_fat in [0.4, 1.8] range, 1.0 = average.
# Values tuned visually against reference character art. Not architectural.
_BUILD_KEYWORDS: dict[str, tuple[float, float, BodyType]] = {
    # keyword → (muscle_mass, body_fat, body_type)
    "muscular": (1.4, 0.7, BodyType.MESOMORPH),
    "athletic": (1.3, 0.6, BodyType.MESOMORPH),
    "strong": (1.3, 0.8, BodyType.MESOMORPH),
    "buff": (1.5, 0.7, BodyType.MESOMORPH),
    "bodybuilder": (1.7, 0.6, BodyType.MESOMORPH),
    "slim": (0.7, 0.5, BodyType.ECTOMORPH),
    "slender": (0.6, 0.5, BodyType.ECTOMORPH),
    "thin": (0.6, 0.4, BodyType.ECTOMORPH),
    "skinny": (0.5, 0.4, BodyType.ECTOMORPH),
    "lean": (0.9, 0.5, BodyType.ECTOMORPH),
    "stocky": (1.1, 1.3, BodyType.ENDOMORPH),
    "heavy": (0.9, 1.4, BodyType.ENDOMORPH),
    "overweight": (0.8, 1.5, BodyType.ENDOMORPH),
    "chubby": (0.7, 1.4, BodyType.ENDOMORPH),
    "fat": (0.7, 1.6, BodyType.ENDOMORPH),
    "obese": (0.8, 1.8, BodyType.ENDOMORPH),
    "plump": (0.7, 1.3, BodyType.ENDOMORPH),
    "broad": (1.2, 1.0, BodyType.MESOMORPH),
    "bulky": (1.4, 1.1, BodyType.ENDOMORPH),
}
_GENDER_KEYWORDS: dict[str, str] = {
    "man": "male", "woman": "female", "boy": "male", "girl": "female",
    "male": "male", "female": "female", "gentleman": "male", "lady": "female",
    "guy": "male", "gal": "female", "he": "male", "she": "female",
    "father": "male", "mother": "female", "husband": "male", "wife": "female",
    "brother": "male", "sister": "female", "son": "male", "daughter": "female",
}
# DESIGN CHOICES — age-based height from WHO growth charts:
# child ~8yo ≈ 1.20m, teen ~14yo ≈ 1.60m, elderly (osteoporosis) ≈ 1.68m.
# Not architectural.
_AGE_KEYWORDS: dict[str, tuple[str, float]] = {
    # keyword → (age_group, height_adjustment)
    "child": ("child", 1.20), "kid": ("child", 1.20),
    "teenager": ("teen", 1.60), "teen": ("teen", 1.60),
    "elderly": ("elderly", 1.68), "old": ("elderly", 1.68),
    "senior": ("elderly", 1.68), "young": ("adult", 1.72),
}


def body_params_from_prompt(prompt: str, entity_name: str = "") -> BodyParams:
    """Extract body shape parameters from natural-language description.

    Scans *prompt* and *entity_name* for height, build, gender, and age
    keywords.  Unspecified parameters keep their defaults.

    Examples
    --------
    >>> bp = body_params_from_prompt("a tall muscular man walks")
    >>> bp.height_m, bp.muscle_mass, bp.gender
    (1.85, 1.4, 'male')

    >>> bp = body_params_from_prompt("a short chubby old woman dances")
    >>> bp.height_m, bp.body_fat, bp.age_group, bp.gender
    (1.60, 1.4, 'elderly', 'female')
    """
    text = f"{prompt} {entity_name}".lower()
    params = BodyParams()
    _match_height_and_build(params, text)
    _match_gender_and_age(params, text)

    params.apply_gender_defaults()
    params.apply_body_type_defaults()
    params.apply_age_defaults()

    log.debug("BodyParams from prompt: height=%.2fm, mass=%.0fkg, "
              "gender=%s, body_type=%s, muscle=%.1f, fat=%.1f",
              params.height_m, params.mass_kg, params.gender,
              params.body_type.name, params.muscle_mass, params.body_fat)
    return params


def _match_height_and_build(params: BodyParams, text: str) -> None:
    """Set height and build keywords on *params* from *text*."""
    for kw in sorted(_HEIGHT_KEYWORDS, key=len, reverse=True):
        if kw in text:
            params.height_m = _HEIGHT_KEYWORDS[kw]
            break
    for kw in sorted(_BUILD_KEYWORDS, key=len, reverse=True):
        if kw in text:
            muscle, fat, btype = _BUILD_KEYWORDS[kw]
            params.muscle_mass = muscle
            params.body_fat = fat
            params.body_type = btype
            break


def _match_gender_and_age(params: BodyParams, text: str) -> None:
    """Set gender and age keywords on *params* from *text*."""
    for kw, gender in _GENDER_KEYWORDS.items():
        if kw in text.split():
            params.gender = gender
            break
    for kw in sorted(_AGE_KEYWORDS, key=len, reverse=True):
        if kw in text:
            age_group, height_adj = _AGE_KEYWORDS[kw]
            params.age_group = age_group
            if params.height_m == 1.72:
                params.height_m = height_adj
            break


# ── Limb cross-section widths ────────────────────────────────────────────────

# Anatomical cross-section widths as fraction of body height.
# Source: DeSilva anthropometric data, scaled by muscle_mass and body_fat.
# These define the "thickness" of each body segment for silhouette generation.

@dataclass(slots=True)
class _LimbWidths:
    """Per-segment pixel widths for the current body + projection scale."""
    head_rx: float = 0.0      # head ellipse semi-axis X (pixels)
    head_ry: float = 0.0      # head ellipse semi-axis Y (pixels)
    neck: float = 0.0
    shoulder: float = 0.0     # torso width at shoulders
    chest: float = 0.0        # torso width at chest level
    waist: float = 0.0        # torso width at waist
    hip: float = 0.0          # torso width at hips
    upper_arm: float = 0.0
    forearm: float = 0.0
    hand: float = 0.0
    upper_leg: float = 0.0
    lower_leg: float = 0.0
    foot_length: float = 0.0
    foot_height: float = 0.0


def _compute_limb_widths(params: BodyParams, px_per_m: float) -> _LimbWidths:
    """Compute pixel-space limb widths from body parameters.

    The widths are proportional to height (via *px_per_m*) and modulated
    by ``muscle_mass`` and ``body_fat``.  This gives a consistent body
    silhouette that responds to prompt descriptors like "muscular" or "slim".
    """
    bw = _base_widths_m(params)
    s = px_per_m * 0.5
    return _LimbWidths(
        head_rx=bw[0] * s, head_ry=bw[1] * s,
        neck=bw[2] * s, shoulder=bw[3] * s,
        chest=bw[4] * s, waist=bw[5] * s, hip=bw[6] * s,
        upper_arm=bw[7] * s, forearm=bw[8] * s, hand=bw[9] * s,
        upper_leg=bw[10] * s, lower_leg=bw[11] * s,
        foot_length=bw[12] * px_per_m, foot_height=bw[13] * px_per_m,
    )


def _base_widths_m(params: BodyParams) -> tuple[float, ...]:
    """Anthropometric base widths in metres, modulated by body descriptors.

    DESIGN CHOICES — multipliers as fraction of body height (not architectural):
    Source: DeSilva (1999) anthropometric tables, cross-referenced with
    NASA STD-3000 ergonomic data. Approximate — sufficient for silhouette
    generation, not accurate enough for biomechanical modelling.
    gf=0.95 for female: women's segment cross-sections average ~5% narrower
    than men's (DeSilva Table 8.6).
    """
    h = params.height_m
    bulk = params.bulk_factor
    gf = 0.95 if params.gender == "female" else 1.0
    return (
        h * 0.085 * params.head_scale,                        # head_w  (8.5% of height)
        h * 0.110 * params.head_scale,                        # head_h  (11% of height)
        h * 0.050 * bulk * gf,                                # neck    (5% of height)
        h * 0.260 * params.shoulder_width * bulk,             # shoulder (26% — biacromial breadth)
        h * 0.180 * bulk * gf,                                # chest   (18%)
        h * 0.145 * (0.7 + 0.3 * params.body_fat),           # waist   (14.5%, fat-sensitive)
        h * 0.195 * params.hip_width * (0.8 + 0.2 * params.body_fat),  # hip (19.5%)
        h * 0.048 * (0.6 + 0.4 * params.muscle_mass),        # upper_arm (4.8%, muscle-sensitive)
        h * 0.038 * (0.7 + 0.3 * params.muscle_mass),        # forearm (3.8%)
        h * 0.030,                                            # hand    (3%)
        h * 0.072 * bulk * gf,                                # upper_leg (7.2%)
        h * 0.052 * (0.7 + 0.3 * params.muscle_mass),        # lower_leg (5.2%)
        h * 0.152,                                            # foot_length (15.2%)
        h * 0.040,                                            # foot_height (4%)
    )


# ── Silhouette polygon generation ───────────────────────────────────────────

def _joint_2d(uvd: np.ndarray, idx: int) -> tuple[float, float]:
    """Extract (x, y) pixel coords for joint *idx* from projected (21,3) array."""
    return float(uvd[idx, 0]), float(uvd[idx, 1])


def _perp_unit(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    """Unit vector perpendicular to the segment p1→p2 (screen-space)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = max(np.sqrt(dx * dx + dy * dy), 1e-6)
    return (-dy / length, dx / length)


def _limb_quad(
    p1: tuple[float, float],
    p2: tuple[float, float],
    w1: float,
    w2: float,
) -> np.ndarray:
    """Generate a filled quadrilateral (trapezoid) for a limb segment.

    The quad is centred on the segment p1→p2 with half-widths w1 at p1
    and w2 at p2.  Returns (4, 2) int32 polygon vertices.
    """
    nx, ny = _perp_unit(p1, p2)
    return np.array([
        [p1[0] + nx * w1, p1[1] + ny * w1],
        [p2[0] + nx * w2, p2[1] + ny * w2],
        [p2[0] - nx * w2, p2[1] - ny * w2],
        [p1[0] - nx * w1, p1[1] - ny * w1],
    ], dtype=np.int32)


def _torso_polygon(
    uvd: np.ndarray,
    widths: _LimbWidths,
) -> np.ndarray:
    """Generate filled torso contour from shoulder ↔ hip joints.

    Creates a smooth 8-point polygon:
    left-shoulder → left-armpit → left-waist → left-hip →
    right-hip → right-waist → right-armpit → right-shoulder
    """
    l_sho = _joint_2d(uvd, L_SHO)
    r_sho = _joint_2d(uvd, R_SHO)
    l_hip = _joint_2d(uvd, L_HIP)
    r_hip = _joint_2d(uvd, R_HIP)
    spine = _joint_2d(uvd, SPINE2)
    waist = _joint_2d(uvd, SPINE1)

    # Perpendicular directions at different torso levels
    nx_sho, ny_sho = _perp_unit(l_sho, r_sho)
    nx_hip, ny_hip = _perp_unit(l_hip, r_hip)

    # Shoulder midpoint
    mid_sho = ((l_sho[0] + r_sho[0]) / 2, (l_sho[1] + r_sho[1]) / 2)
    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

    # Waist point (between spine1 and hips)
    waist_pt = ((spine[0] + waist[0]) / 2, (spine[1] + waist[1]) / 2)
    nx_w, ny_w = _perp_unit(mid_sho, mid_hip)

    return np.array([
        # Left side (top to bottom)
        [mid_sho[0] + nx_sho * widths.shoulder, mid_sho[1] + ny_sho * widths.shoulder],
        [spine[0] + nx_w * widths.chest, spine[1] + ny_w * widths.chest],
        [waist_pt[0] + nx_w * widths.waist, waist_pt[1] + ny_w * widths.waist],
        [mid_hip[0] + nx_hip * widths.hip, mid_hip[1] + ny_hip * widths.hip],
        # Right side (bottom to top)
        [mid_hip[0] - nx_hip * widths.hip, mid_hip[1] - ny_hip * widths.hip],
        [waist_pt[0] - nx_w * widths.waist, waist_pt[1] - ny_w * widths.waist],
        [spine[0] - nx_w * widths.chest, spine[1] - ny_w * widths.chest],
        [mid_sho[0] - nx_sho * widths.shoulder, mid_sho[1] - ny_sho * widths.shoulder],
    ], dtype=np.int32)


def _draw_head_neck(mask: np.ndarray, uvd: np.ndarray, widths: _LimbWidths) -> None:
    """Draw head ellipse and neck quad onto the body mask."""
    neck_top = _joint_2d(uvd, NECK)
    neck_base = _joint_2d(uvd, SPINE2)
    neck_quad = _limb_quad(neck_top, neck_base, widths.neck, widths.neck * 1.2)
    cv2.fillPoly(mask, [neck_quad], (255,))

    head_pt = _joint_2d(uvd, HEAD)
    head_center = (int(head_pt[0]), int(head_pt[1]))
    rx = max(4, int(widths.head_rx))
    ry = max(5, int(widths.head_ry))
    cv2.ellipse(mask, head_center, (rx, ry), 0, 0, 360, (255,), -1)


def _draw_arm(mask: np.ndarray, uvd: np.ndarray, widths: _LimbWidths,
              sho_idx: int, elb_idx: int, wri_idx: int) -> None:
    """Draw one arm (upper arm + forearm + hand) onto the body mask."""
    sho = _joint_2d(uvd, sho_idx)
    elb = _joint_2d(uvd, elb_idx)
    wri = _joint_2d(uvd, wri_idx)

    ua = _limb_quad(sho, elb, widths.upper_arm, widths.forearm * 1.05)
    cv2.fillPoly(mask, [ua], (255,))

    fa = _limb_quad(elb, wri, widths.forearm, widths.hand * 0.9)
    cv2.fillPoly(mask, [fa], (255,))

    cv2.circle(mask, (int(elb[0]), int(elb[1])),
               max(3, int(widths.forearm * 1.05)), (255,), -1)
    cv2.circle(mask, (int(sho[0]), int(sho[1])),
               max(3, int(widths.upper_arm)), (255,), -1)
    cv2.circle(mask, (int(wri[0]), int(wri[1])),
               max(2, int(widths.hand)), (255,), -1)


def _draw_leg(mask: np.ndarray, uvd: np.ndarray, widths: _LimbWidths,
              hip_idx: int, kne_idx: int, ank_idx: int,
              heel_idx: int, toe_idx: int) -> None:
    """Draw one leg (upper + lower + foot) onto the body mask."""
    hip = _joint_2d(uvd, hip_idx)
    kne = _joint_2d(uvd, kne_idx)
    ank = _joint_2d(uvd, ank_idx)
    heel = _joint_2d(uvd, heel_idx)
    toe = _joint_2d(uvd, toe_idx)

    ul = _limb_quad(hip, kne, widths.upper_leg, widths.lower_leg * 1.1)
    cv2.fillPoly(mask, [ul], (255,))

    ll = _limb_quad(kne, ank, widths.lower_leg, widths.lower_leg * 0.65)
    cv2.fillPoly(mask, [ll], (255,))

    cv2.circle(mask, (int(kne[0]), int(kne[1])),
               max(3, int(widths.lower_leg * 1.1)), (255,), -1)
    cv2.circle(mask, (int(hip[0]), int(hip[1])),
               max(3, int(widths.upper_leg)), (255,), -1)

    foot_w = max(2, int(widths.foot_height * 0.5))
    foot_poly = _limb_quad(heel, toe, foot_w, foot_w * 0.7)
    cv2.fillPoly(mask, [foot_poly], (255,))


def _build_body_mask(uvd: np.ndarray, widths: _LimbWidths, h: int, w: int) -> np.ndarray:
    """Build a complete body silhouette mask from joints and limb widths."""
    mask = np.zeros((h, w), dtype=np.uint8)

    torso = _torso_polygon(uvd, widths)
    cv2.fillPoly(mask, [torso], (255,))

    _draw_head_neck(mask, uvd, widths)
    _draw_arm(mask, uvd, widths, L_SHO, L_ELB, L_WRI)
    _draw_arm(mask, uvd, widths, R_SHO, R_ELB, R_WRI)
    _draw_leg(mask, uvd, widths, L_HIP, L_KNE, L_ANK, L_HEEL, L_TOE)
    _draw_leg(mask, uvd, widths, R_HIP, R_KNE, R_ANK, R_HEEL, R_TOE)

    return cv2.GaussianBlur(mask, (3, 3), sigmaX=0.8)


def _composite_silhouette(
    out: np.ndarray, mask: np.ndarray,
    color: tuple[int, int, int],
    outline_color: tuple[int, int, int],
    outline_thickness: int,
) -> np.ndarray:
    """Blend silhouette colour onto canvas and draw outline."""
    alpha = mask.astype(np.float32) / 255.0
    color_f = np.array(color, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - alpha) + color_f[c] * alpha

    if outline_thickness > 0:
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, outline_color,
                         outline_thickness, cv2.LINE_AA)

    return np.clip(out, 0, 255).astype(np.uint8)


def render_body_silhouette(
    uvd: np.ndarray,
    params: BodyParams,
    canvas: np.ndarray,
    px_per_m: float,
    *,
    fill_color: tuple[int, int, int] | None = None,
    outline_color: tuple[int, int, int] = (40, 35, 30),
    outline_thickness: int = 1,
    shadow: bool = True,
) -> np.ndarray:
    """Render a filled human body silhouette onto *canvas*."""
    h, w = canvas.shape[:2]
    widths = _compute_limb_widths(params, px_per_m)
    color = fill_color or params.skin_tone

    out = canvas.astype(np.float32)
    if shadow:
        _draw_ground_shadow(out, uvd, widths, w, h)

    mask = _build_body_mask(uvd, widths, h, w)
    return _composite_silhouette(out, mask, color, outline_color, outline_thickness)


def _draw_ground_shadow(
    canvas: np.ndarray,
    uvd: np.ndarray,
    widths: _LimbWidths,
    w: int,
    h: int,
) -> None:
    """Draw a soft elliptical shadow under the feet."""
    foot_joints = [L_ANK, L_HEEL, L_TOE, R_ANK, R_HEEL, R_TOE]
    foot_xs = [float(uvd[j, 0]) for j in foot_joints]
    foot_ys = [float(uvd[j, 1]) for j in foot_joints]

    cx = int(np.clip(np.mean(foot_xs), 0, w - 1))
    cy = int(np.clip(np.max(foot_ys) + 8, 0, h - 1))
    spread = max(15, int((max(foot_xs) - min(foot_xs)) * 0.7 + widths.foot_length * 0.5))

    shadow = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(shadow, (cx, cy), (spread, max(6, spread // 4)), 0, 0, 360, (0.45,), -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(4, spread * 0.3))
    canvas *= (1.0 - shadow[:, :, np.newaxis] * 0.5)


# ── High-level API ───────────────────────────────────────────────────────────

def estimate_px_per_m(uvd: np.ndarray, body_height_m: float) -> float:
    """Estimate the pixel-per-metre scale from the projected skeleton.

    Uses the vertical extent (head to feet) of the projected skeleton
    as the height reference.
    """
    y_coords = uvd[:, 1]
    skeleton_height_px = float(np.ptp(y_coords))  # peak-to-peak
    if skeleton_height_px < 10:
        return 200.0  # fallback
    return skeleton_height_px / max(body_height_m, 0.5)
