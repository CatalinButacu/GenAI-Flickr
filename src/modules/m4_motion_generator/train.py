"""
#WHERE
    Re-export shim — keeps existing `from .train import X` working.
    New code should import from nn_models, dataset, or trainer directly.

#WHAT
    Backward-compatible re-export of M4 training components.

#INPUT / #OUTPUT
    See nn_models.py, dataset.py, trainer.py for details.
"""

from .nn_models import SimpleTextEncoder, MotionDecoder, TextToMotionSSM  # noqa: F401
from .dataset import KITMLDatasetTorch                                    # noqa: F401
from .trainer import TrainingConfig, Trainer, train_motion_ssm            # noqa: F401
