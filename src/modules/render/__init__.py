from .engine import RenderEngine, RenderSettings
from .aitviewer_renderer import AitviewerRenderer


def get_best_renderer(img_w: int = 1280, img_h: int = 720):
    """Return the aitviewer-based renderer."""
    return AitviewerRenderer(img_w, img_h)


__all__ = [
    "RenderEngine", "RenderSettings",
    "AitviewerRenderer",
    "get_best_renderer",
]
