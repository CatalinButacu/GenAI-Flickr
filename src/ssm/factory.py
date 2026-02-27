"""
#WHERE
    Used by src/ssm/__init__.py as the main entry point for creating
    SSM layers programmatically.

#WHAT
    Factory functions for SSM layer instantiation and module introspection.

#INPUT
    layer_type string ("s4", "mamba", "temporal", "motion", "physics")
    plus keyword arguments forwarded to the chosen layer constructor.

#OUTPUT
    An instantiated SSM layer, or a dict of module capabilities.
"""

import logging

from .config import SSMConfig
from .s4 import S4Layer
from .mamba import MambaLayer
from .composites import TemporalConsistencySSM, MotionSSM, PhysicsSSM

logger = logging.getLogger(__name__)

_LAYER_BUILDERS = {
    "s4":       lambda **kw: S4Layer(**kw),
    "mamba":    lambda **kw: MambaLayer(SSMConfig(**kw)),
    "temporal": lambda **kw: TemporalConsistencySSM(**kw),
    "motion":   lambda **kw: MotionSSM(**kw),
    "physics":  lambda **kw: PhysicsSSM(**kw),
}


def create_ssm_layer(layer_type: str = "mamba", **kwargs):
    """Factory: create an SSM layer by name."""
    builder = _LAYER_BUILDERS.get(layer_type)
    if builder is None:
        raise ValueError(f"Unknown layer type: {layer_type}")
    return builder(**kwargs)


def get_ssm_info() -> dict:
    """Module capabilities and references."""
    return {
        "torch_available": True,
        "layers": [
            "S4Layer", "MambaLayer", "TemporalConsistencySSM",
            "MotionSSM", "PhysicsSSM",
        ],
        "references": {
            "S4": "https://arxiv.org/abs/2111.00396",
            "Mamba": "https://arxiv.org/abs/2312.00752",
            "Motion Mamba": "https://arxiv.org/abs/2403.07487",
            "HiPPO": "https://arxiv.org/abs/2008.07669",
        },
        "novel_contribution": "PhysicsSSM — Physics-Constrained State Space Model",
    }
