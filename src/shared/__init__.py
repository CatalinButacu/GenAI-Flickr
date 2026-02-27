"""
#WHERE
    Imported by every pipeline module (M1–M8) and by tests/benchmarks.

#WHAT
    Shared vocabulary, action/object registries, color and spatial enums.

#INPUT
    None (constant registries).

#OUTPUT
    ACTIONS, OBJECTS dictionaries; ActionDefinition, ObjectDefinition dataclasses;
    ColorName, SpatialRelation enums; lookup helpers.
"""

from .vocabulary import (
    # Actions
    ACTIONS,
    ActionDefinition,
    ActionCategory,
    get_action_by_keyword,
    get_action_names,
    get_rl_action_space_size,
    
    # Objects
    OBJECTS,
    ObjectDefinition,
    ObjectCategory,
    get_object_by_keyword,
    get_object_names,
    
    # Properties
    ColorName,
    COLOR_KEYWORDS,
    SpatialRelation,
    SPATIAL_KEYWORDS,
)

__all__ = [
    "ACTIONS",
    "ActionDefinition", 
    "ActionCategory",
    "get_action_by_keyword",
    "get_action_names",
    "get_rl_action_space_size",
    "OBJECTS",
    "ObjectDefinition",
    "ObjectCategory",
    "get_object_by_keyword",
    "get_object_names",
    "ColorName",
    "COLOR_KEYWORDS",
    "SpatialRelation",
    "SPATIAL_KEYWORDS",
]
