"""
Render Engine — Module 7
========================
Cinematic post-processing: motion blur, DoF, color grade, vignette, film grain.

Quick start::

    from src.modules.m7_render_engine import RenderEngine, RenderSettings

    engine = RenderEngine(RenderSettings(motion_blur=True, dof=True, vignette=True))
    engine.render(frames, "outputs/videos/output.mp4", fps=24)
"""
from .render_engine import RenderEngine, RenderSettings

__all__ = ["RenderEngine", "RenderSettings"]
