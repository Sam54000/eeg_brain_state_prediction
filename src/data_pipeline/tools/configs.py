import numpy as np
import scipy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import os

@dataclass
class PipelineConfig:
    """Configuration class for the pipeline"""
    n_threads: int = 32
    data_root: Path = Path("/data2/Projects/eeg_fmri_natview/derivatives")
    overwrite: bool = False
    code_root: Path = Path(
        os.environ["HOME"],
        "01_projects",
        "eeg_brain_state_prediction",
    )
    tasks: list[str] = field(default_factory=lambda: ["rest"])

@dataclass
class MultimodalConfig:
    """Configuration class for model parameters"""
    resampling_factor: int = 8
    sampling_rate_hz: float = 3.8
    tr_time_seconds: float = 2.1

@dataclass
class BrainstatesConfig:
    description: list[str] = field(default_factory=lambda: ["Cpca1054"])
    brainstates: np.ndarray = field(default_factory=lambda: np.array([
        'CAP1', 'CAP2', 'CAP3', 'CAP4', 'CAP5', 'CAP6', 'CAP7', 'CAP8'
    ]))

@dataclass
class EegConfig:
    """Configuration class for EEG data"""
    description: str = "RawBk"
    sampling_rate_hz: float = 200
    montage: str = "easycap-M1"
    low_frequency_hz: float = 0.5
    high_frequency_hz: float = 40
    channels: Optional[list[str]] = None
    tmin: Optional[float] = None
    tmax: Optional[float] = None

@dataclass
class EegFeaturesConfig:
    """Configuration class for EEG features"""
    frequencies: list[tuple[float, float]] = field(default_factory=lambda: [(0.5, 40)])
    

    