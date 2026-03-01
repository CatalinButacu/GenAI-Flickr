"""
#WHERE
    Imported by pipeline.py, demo scripts.

#WHAT
    Render Engine Module (Module 7) — cinematic post-processing: motion
    blur, depth of field, color grading, vignette, film grain.

#INPUT
    List[FrameData] from M5, RenderSettings.

#OUTPUT
    Post-processed MP4 video file.
"""
from .render_engine import RenderEngine, RenderSettings

__all__ = ["RenderEngine", "RenderSettings"]
