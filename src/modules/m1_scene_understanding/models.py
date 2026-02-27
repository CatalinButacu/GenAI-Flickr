"""
#WHERE
    Re-export shim — keeps existing `from .models import X` working.
    New code should import from extraction_models or scene_models directly.

#WHAT
    Backward-compatible re-export of all M1 data models.

#INPUT / #OUTPUT
    See the individual sub-modules for details.
"""

# Extraction + reasoning stage
from .extraction_models import (          # noqa: F401
    EntityType,
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedAction,
    ExtractedRelation,
    ExtractionResult,
    ImplicitEntity,
    SpatialHint,
    ActivityTemplate,
)

# Final scene description
from .scene_models import (               # noqa: F401
    SceneObject,
    SceneAction,
    CameraMotion,
    SceneDescription,
)
