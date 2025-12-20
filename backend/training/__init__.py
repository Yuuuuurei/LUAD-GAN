"""
Training module for GAN.
"""

from .trainer import WGANTrainer
from .losses import WassersteinLoss, VanillaGANLoss
from .callbacks import EarlyStopping, ModelCheckpoint, LossHistory, ProgressCallback

__all__ = [
    'WGANTrainer',
    'WassersteinLoss',
    'VanillaGANLoss',
    'EarlyStopping',
    'ModelCheckpoint',
    'LossHistory',
    'ProgressCallback'
]