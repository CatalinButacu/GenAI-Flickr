"""
#WHERE
    Imported by pipeline.py, benchmarks, test_modules.py.

#WHAT
    Scene Understanding Module (Module 1) — NLP text parsing via T5 or
    rule-based parser, knowledge retrieval, and scene description building.

#INPUT
    Natural language text prompt.

#OUTPUT
    ParsedScene / SceneDescription with entities, actions, relations.
"""

from .orchestrator import StoryAgent
from .prompt_parser import PromptParser, ParsedScene, ParsedEntity, ParsedAction
from .t5_parser import T5SceneParser
from .models import CameraMotion, SceneAction, SceneDescription, SceneObject

__all__ = [
    "PromptParser", "T5SceneParser", "ParsedScene", "ParsedEntity", "ParsedAction",
    "StoryAgent", "SceneDescription", "SceneObject", "SceneAction", "CameraMotion",
]
