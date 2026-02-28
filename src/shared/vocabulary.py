"""
#WHERE
    Imported by shared/__init__.py → re-exported to all pipeline modules,
    planner.py, builder.py, prompt_parser.py, ssm_generator.py, physics.py.

#WHAT
    Single source of truth for action verbs, object types, colors, and
    spatial relations used across the entire pipeline.

#INPUT
    None (constant definitions).

#OUTPUT
    ACTIONS dict, OBJECTS dict, ActionCategory/ObjectCategory/ColorName/
    SpatialRelation enums, lookup helpers.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class ActionCategory(Enum):
    LOCOMOTION   = auto()
    MANIPULATION = auto()
    INTERACTION  = auto()
    GESTURE      = auto()
    POSE         = auto()
    PHYSICS      = auto()


@dataclass(slots=True)
class ActionDefinition:
    """Single action entry."""
    name: str
    category: ActionCategory
    keywords: list[str]       = field(default_factory=list)
    requires_target: bool     = False
    motion_clip: Optional[str] = None
    rl_action_id: Optional[int] = None


ACTIONS: dict[str, ActionDefinition] = {
    "walk":    ActionDefinition("walk",    ActionCategory.LOCOMOTION,   ["walk", "walking", "walks", "stroll", "move", "go"],                        True,  "walk_forward",  0),
    "run":     ActionDefinition("run",     ActionCategory.LOCOMOTION,   ["run", "running", "runs", "sprint", "dash", "rush"],                        True,  "run_forward",   1),
    "jump":    ActionDefinition("jump",    ActionCategory.LOCOMOTION,   ["jump", "jumping", "jumps", "leap", "hop"],                                 False, "jump_in_place", 2),
    "stand":   ActionDefinition("stand",   ActionCategory.POSE,         ["stand", "standing", "stands", "stop", "halt", "wait"],                    False, "idle_stand",    3),
    "pick_up": ActionDefinition("pick_up", ActionCategory.MANIPULATION, ["pick", "pick up", "picks up", "picking up", "picks", "grab", "grabs", "take", "takes", "lift", "lifts", "grasp"],                       True,  "pick_object",   4),
    "throw":   ActionDefinition("throw",   ActionCategory.MANIPULATION, ["throw", "throwing", "throws", "toss", "hurl"],                            True,  "throw_object",  5),
    "place":   ActionDefinition("place",   ActionCategory.MANIPULATION, ["place", "put", "put down", "set", "drop", "release"],                    True,  "place_object",  6),
    "kick":    ActionDefinition("kick",    ActionCategory.INTERACTION,  ["kick", "kicking", "kicks", "boot", "punt"],                               True,  "kick_object",   7),
    "push":    ActionDefinition("push",    ActionCategory.INTERACTION,  ["push", "pushing", "pushes", "shove"],                                     True,  "push_object",   8),
    "wave":    ActionDefinition("wave",    ActionCategory.GESTURE,      ["wave", "waving", "waves", "greet", "hello"],                              False, "wave_hand",     9),
    "fall":    ActionDefinition("fall",    ActionCategory.PHYSICS,      ["fall", "falls", "falling", "drop", "drops", "dropping"],                  True,  None,            None),
    "roll":    ActionDefinition("roll",    ActionCategory.PHYSICS,      ["roll", "rolls", "rolling"],                                               False, None,            None),
    "bounce":  ActionDefinition("bounce",  ActionCategory.PHYSICS,      ["bounce", "bounces", "bouncing", "rebound"],                               False, None,            None),
    "slide":   ActionDefinition("slide",   ActionCategory.PHYSICS,      ["slide", "slides", "sliding", "glide"],                                   False, None,            None),
    "collide": ActionDefinition("collide", ActionCategory.PHYSICS,      ["collide", "collides", "hit", "hits", "crash", "crashes", "impact"],       True,  None,            None),
}


class ObjectCategory(Enum):
    PRIMITIVE = auto()
    FURNITURE = auto()
    SPORTS    = auto()
    CONTAINER = auto()
    HUMANOID  = auto()


@dataclass(slots=True)
class ObjectDefinition:
    """Single object entry."""
    name: str
    category: ObjectCategory
    keywords: list[str]          = field(default_factory=list)
    default_shape: str           = "box"
    default_size: list[float]    = field(default_factory=lambda: [0.1, 0.1, 0.1])
    default_mass: float          = 1.0
    can_be_grasped: bool         = True
    mesh_prompt: Optional[str]   = None


OBJECTS: dict[str, ObjectDefinition] = {
    "sphere":   ObjectDefinition("sphere",   ObjectCategory.PRIMITIVE, ["sphere", "ball", "orb", "globe"],                                                          "sphere",   [0.1],            0.5,  True,  None),
    "cube":     ObjectDefinition("cube",     ObjectCategory.PRIMITIVE, ["cube", "box", "block", "crate"],                                                           "box",      [0.1, 0.1, 0.1], 1.0,  True,  None),
    "cylinder": ObjectDefinition("cylinder", ObjectCategory.PRIMITIVE, ["cylinder", "tube", "pipe", "rod", "pillar"],                                               "cylinder", [0.05, 0.2],      0.8,  True,  None),
    "football": ObjectDefinition("football", ObjectCategory.SPORTS,    ["football", "soccer ball"],                                                                 "sphere",   [0.11],           0.43, True,  "a white and black soccer football"),
    "table":    ObjectDefinition("table",    ObjectCategory.FURNITURE, ["table", "desk", "surface"],                                                                "box",      [0.8, 0.5, 0.02], 0.0,  False, None),
    "humanoid": ObjectDefinition("humanoid", ObjectCategory.HUMANOID,  ["person", "human", "man", "woman", "character", "robot", "figure", "agent", "actor"],      "humanoid", [1.7],            70.0, False, None),
}


class ColorName(Enum):
    RED    = (1.0, 0.0, 0.0, 1.0)
    GREEN  = (0.0, 1.0, 0.0, 1.0)
    BLUE   = (0.0, 0.0, 1.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0, 1.0)
    ORANGE = (1.0, 0.5, 0.0, 1.0)
    PURPLE = (0.5, 0.0, 0.5, 1.0)
    WHITE  = (1.0, 1.0, 1.0, 1.0)
    BLACK  = (0.1, 0.1, 0.1, 1.0)
    GRAY   = (0.5, 0.5, 0.5, 1.0)
    BROWN  = (0.6, 0.3, 0.1, 1.0)


COLOR_KEYWORDS: dict[str, ColorName] = {
    "red":    ColorName.RED,    "green":  ColorName.GREEN,  "blue":   ColorName.BLUE,
    "yellow": ColorName.YELLOW, "orange": ColorName.ORANGE,
    "purple": ColorName.PURPLE, "violet": ColorName.PURPLE,
    "white":  ColorName.WHITE,  "black":  ColorName.BLACK,
    "gray":   ColorName.GRAY,   "grey":   ColorName.GRAY,   "brown":  ColorName.BROWN,
}


class SpatialRelation(Enum):
    ON       = auto()
    UNDER    = auto()
    NEXT_TO  = auto()
    IN_FRONT = auto()
    BEHIND   = auto()
    LEFT_OF  = auto()
    RIGHT_OF = auto()
    NEAR     = auto()
    FAR      = auto()


SPATIAL_KEYWORDS: dict[str, SpatialRelation] = {
    "on":           SpatialRelation.ON,
    "on top of":    SpatialRelation.ON,
    "above":        SpatialRelation.ON,
    "under":        SpatialRelation.UNDER,
    "below":        SpatialRelation.UNDER,
    "beneath":      SpatialRelation.UNDER,
    "next to":      SpatialRelation.NEXT_TO,
    "beside":       SpatialRelation.NEXT_TO,
    "near":         SpatialRelation.NEAR,
    "close to":     SpatialRelation.NEAR,
    "in front of":  SpatialRelation.IN_FRONT,
    "behind":       SpatialRelation.BEHIND,
    "left of":      SpatialRelation.LEFT_OF,
    "right of":     SpatialRelation.RIGHT_OF,
    "far from":     SpatialRelation.FAR,
}


def get_action_by_keyword(keyword: str) -> Optional[ActionDefinition]:
    """Return first ActionDefinition whose keywords include the given word."""
    kw = keyword.lower()
    return next((a for a in ACTIONS.values() if kw in a.keywords), None)


def get_object_by_keyword(keyword: str) -> Optional[ObjectDefinition]:
    """Return first ObjectDefinition whose keywords include the given word."""
    kw = keyword.lower()
    return next((o for o in OBJECTS.values() if kw in o.keywords), None)


def get_rl_action_space_size() -> int:
    return sum(1 for a in ACTIONS.values() if a.rl_action_id is not None)


def get_action_names() -> list[str]:
    return list(ACTIONS.keys())


def get_object_names() -> list[str]:
    return list(OBJECTS.keys())
