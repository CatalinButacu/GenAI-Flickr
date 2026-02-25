"""M7: Post-processing renderer. Stub — not yet implemented.

Video export is currently delegated to M5 Simulator.create_video().
Plan: motion blur, DOF, color grading, HDRI compositing.
"""
from typing import List


class RenderEngine:
    def setup(self) -> None:
        raise NotImplementedError("M7 RenderEngine not yet implemented")

    def render(self, frames: List, output_path: str) -> str:
        raise NotImplementedError("M7 RenderEngine not yet implemented")
