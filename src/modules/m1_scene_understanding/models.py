"""
#WHERE
    Re-export shim — keeps existing `from .models import X` working.
    New code should import from extraction_models, scene_models,
    or reasoning_models directly.

#WHAT
    Backward-compatible re-export of all M1 data models.

#INPUT / #OUTPUT
    See the individual sub-modules for details.
"""

# Extraction stage
from .extraction_models import (          # noqa: F401
    EntityType,
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedAction,
    ExtractedRelation,
    ExtractionResult,
)

# Final scene description
from .scene_models import (               # noqa: F401
    SceneObject,
    SceneAction,
    CameraMotion,
    SceneDescription,
)

# Reasoning / templates
from .reasoning_models import (           # noqa: F401
    ImplicitEntity,
    SpatialHint,
    ActivityTemplate,
)
