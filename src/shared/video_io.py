"""
#WHERE
    Used by M5 (simulator), M7 (render engine), M8 (AI enhancer),
    and pipeline.py for writing output videos.

#WHAT
    Thin wrapper around imageio for H.264 video writing.
    Centralises the codec/fps/makedirs boilerplate that was duplicated
    in three modules.

#INPUT
    List of numpy uint8 frames, output path, fps.

#OUTPUT
    MP4 file on disk.
"""

import logging
import os
from typing import List

import numpy as np

log = logging.getLogger(__name__)


def write_video(frames: List[np.ndarray], output_path: str, fps: int = 24) -> str:
    """Write a list of BGR/RGB uint8 frames to an H.264 MP4.

    Returns the absolute path of the written file.
    """
    import imageio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    log.info("Video saved: %s (%d frames, %d fps)", output_path, len(frames), fps)
    return os.path.abspath(output_path)
