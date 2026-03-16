from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from functools import cache


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
    motion_clip: str | None = None


ACTIONS: dict[str, ActionDefinition] = {
    "walk":    ActionDefinition("walk",    ActionCategory.LOCOMOTION,   ["walk", "walking", "walks", "stroll", "move", "go"],                        True,  "walk_forward"),
    "run":     ActionDefinition("run",     ActionCategory.LOCOMOTION,   ["run", "running", "runs", "sprint", "dash", "rush"],                        True,  "run_forward"),
    "jump":    ActionDefinition("jump",    ActionCategory.LOCOMOTION,   ["jump", "jumping", "jumps", "leap", "hop"],                                 False, "jump_in_place"),
    "stand":   ActionDefinition("stand",   ActionCategory.POSE,         ["stand", "standing", "stands", "stop", "halt", "wait"],                    False, "idle_stand"),
    "pick_up": ActionDefinition("pick_up", ActionCategory.MANIPULATION, ["pick", "pick up", "picks up", "picking up", "picks", "grab", "grabs", "take", "takes", "lift", "lifts", "grasp"],                       True,  "pick_object"),
    "throw":   ActionDefinition("throw",   ActionCategory.MANIPULATION, ["throw", "throwing", "throws", "toss", "hurl"],                            True,  "throw_object"),
    "place":   ActionDefinition("place",   ActionCategory.MANIPULATION, ["place", "put", "put down", "set", "drop", "release"],                    True,  "place_object"),
    "kick":    ActionDefinition("kick",    ActionCategory.INTERACTION,  ["kick", "kicking", "kicks", "boot", "punt"],                               True,  "kick_object"),
    "push":    ActionDefinition("push",    ActionCategory.INTERACTION,  ["push", "pushing", "pushes", "shove"],                                     True,  "push_object"),
    "wave":    ActionDefinition("wave",    ActionCategory.GESTURE,      ["wave", "waving", "waves", "greet", "hello"],                              False, "wave_hand"),
    "fall":    ActionDefinition("fall",    ActionCategory.PHYSICS,      ["fall", "falls", "falling", "drop", "drops", "dropping"],                  True,  None),
    "roll":    ActionDefinition("roll",    ActionCategory.PHYSICS,      ["roll", "rolls", "rolling"],                                               False, None),
    "bounce":  ActionDefinition("bounce",  ActionCategory.PHYSICS,      ["bounce", "bounces", "bouncing", "rebound"],                               False, None),
    "slide":   ActionDefinition("slide",   ActionCategory.PHYSICS,      ["slide", "slides", "sliding", "glide"],                                   False, None),
    "collide": ActionDefinition("collide", ActionCategory.PHYSICS,      ["collide", "collides", "hit", "hits", "crash", "crashes", "impact"],       True,  None),
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
    mesh_prompt: str | None   = None


OBJECTS: dict[str, ObjectDefinition] = {
    "sphere":   ObjectDefinition("sphere",   ObjectCategory.PRIMITIVE, ["sphere", "spheres", "ball", "balls", "orb", "orbs", "globe"],                                   "sphere",   [0.1],            0.5,  True,  None),
    "cube":     ObjectDefinition("cube",     ObjectCategory.PRIMITIVE, ["cube", "cubes", "box", "boxes", "block", "blocks", "crate", "crates"],                          "box",      [0.1, 0.1, 0.1], 1.0,  True,  None),
    "cylinder": ObjectDefinition("cylinder", ObjectCategory.PRIMITIVE, ["cylinder", "cylinders", "tube", "tubes", "pipe", "pipes", "rod", "rods", "pillar", "pillars"],  "cylinder", [0.05, 0.2],      0.8,  True,  None),
    "football": ObjectDefinition("football", ObjectCategory.SPORTS,    ["football", "footballs", "soccer ball", "soccer balls"],                                          "sphere",   [0.11],           0.43, True,  "a white and black soccer football"),
    "table":    ObjectDefinition("table",    ObjectCategory.FURNITURE, ["table", "tables", "desk", "desks", "surface", "surfaces"],                                       "box",      [0.8, 0.5, 0.02], 0.0,  False, None),
    "humanoid": ObjectDefinition("humanoid", ObjectCategory.HUMANOID,  ["person", "people", "human", "humans", "man", "men", "woman", "women", "character", "characters", "robot", "robots", "figure", "figures", "agent", "actor"],  "humanoid", [1.7], 70.0, False, None),
}


@cache
def get_action_by_keyword(keyword: str) -> ActionDefinition | None:
    """Return first ActionDefinition whose keywords include the given word."""
    kw = keyword.lower()
    return next((a for a in ACTIONS.values() if kw in a.keywords), None)


@cache
def get_object_by_keyword(keyword: str) -> ObjectDefinition | None:
    """Return first ObjectDefinition whose keywords include the given word."""
    kw = keyword.lower()
    return next((o for o in OBJECTS.values() if kw in o.keywords), None)
