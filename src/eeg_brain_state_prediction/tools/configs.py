import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import scipy.stats

import numpy as np

@dataclass
class PipelineConfig:
    """Configuration class for the pipeline
    
    Attributes:
        n_threads (int): Number of threads to use for parallel processing
        data_root (Path): Root directory for data storage
        overwrite (bool): Whether to overwrite existing files
        code_root (Path): Root directory for code
        tasks (list[str]): List of tasks to process
    """
    n_threads: int = 32
    raw_path: Path = Path("/data2/Projects/eeg_fmri_natview/raw")
    derivatives_path: Path = Path("/data2/Projects/eeg_fmri_natview/derivatives")
    corr_output_path: Path = Path(f"data/eeg_bands_cpca/")
    overwrite: bool = False
    code_root: Path = Path(
        os.environ["HOME"],
        "01_projects",
        "eeg_brain_state_prediction",
    )
    tasks: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    sessions: Optional[List[str]] = None
    runs: Optional[List[str]] = None

@dataclass
class MultimodalConfig:
    """Configuration class for multimodal data processing
    
    Attributes:
        resampling_factor (int): Factor by which to resample the data
        sampling_rate_hz (float): Sampling rate in Hz
        tr_time_seconds (float): TR time in seconds
    """
    resampling_factor: int = 8
    sampling_rate_hz: float = 3.8
    tr_time_seconds: float = 2.1
    modalities: List[str] = field(default_factory=lambda: ["brainstates", "eeg", "eyetracking"])
    brainstates: "BrainstatesConfig" = field(default_factory="BrainstatesConfig")
    eeg: "EegConfig" = field(default_factory="EegConfig")
    eyetracking: "EyeConfig" = field(default_factory="EyeConfig")
    additional_description: str = ""

@dataclass
class EyeConfig:
    features: List[str] = field(default_factory=lambda: ["pupil_dilation"])
    description: Optional[str] = None

@dataclass
class BrainstatesConfig:
    description: list[str] = field(default_factory=lambda: ["Cpca1054"])
    brainstates: np.ndarray = field(default_factory=lambda: np.array([
        'CAP1', 'CAP2', 'CAP3', 'CAP4', 'CAP5', 'CAP6', 'CAP7', 'CAP8'
    ]))

@dataclass
class EegConfig:
    """Configuration class for EEG data processing
    
    Attributes:
        description (str): Description of the EEG data
        sampling_rate_hz (float): Sampling rate in Hz
        montage (str): EEG montage type
        low_frequency_hz (float): Low frequency cutoff for filtering
        high_frequency_hz (float): High frequency cutoff for filtering
        channels (Optional[list[str]]): List of channels to use
        tmin (Optional[float]): Start time for analysis
        tmax (Optional[float]): End time for analysis
    """
    description: str = "RawBk"
    sampling_rate_hz: float = 200
    montage: str = "easycap-M1"
    low_frequency_hz: float = 0.5
    high_frequency_hz: float = 40
    channels: Optional[List[str]] = None
    tmin: Optional[float] = None
    tmax: Optional[float] = None

class EegFeaturesConfig:
    def __init__(self, 
                 frequencies: Optional[list[tuple]],
                 n_channels: int = 1):
        """Configuration class for EEG feature extraction
        
        Attributes:
            frequencies (list[tuple[float, float]]): List of frequency bands to extract
        """
        self.frequencies = frequencies
        self.n_channels = n_channels


@dataclass
class ModelConfig:
    """Configuration class for model parameters"""

    aggregation_function: Callable[[np.ndarray, float], tuple[float, float]] = scipy.stats.ttest_1samp
    stat_func_kwargs: Dict[str, Any] = field(default_factory=lambda: {"popmean": 0})
    nb_desired_features: List[int] = field(
        default_factory=lambda: [ModelConfig.n_bands * ModelConfig.n_channels]
    )
    features_data_filename: Optional[str | Path] = None # This should be defined as a function of the PipelineConfig

    def select_features(self, 
                        eyetracking_features: list[str],
                        n_eeg_channels: int,
                        n_eeg_bands: int
                        ) -> Dict[str, Any]:
        """Select the features to use for the model"""
        feature_set = {
            "eyetracking": 
                eyetracking_features,
            "eeg": {
                "channel": np.arange(n_eeg_channels).repeat(n_eeg_bands),
                "band": np.tile(np.arange(n_eeg_bands),n_eeg_channels),
            }
        } 
        return feature_set
