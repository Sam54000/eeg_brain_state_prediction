"""
EEG Brain State Prediction Data Pipeline

This package contains tools and pipelines for processing EEG data and extracting features
for brain state prediction.
"""

from . import tools
from . import feature_extraction
from . import multimodal

__all__ = ['tools', 'feature_extraction', 'multimodal'] 