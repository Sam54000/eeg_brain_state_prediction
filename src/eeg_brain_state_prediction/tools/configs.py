import os
import sklearn
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
        raw_path (Path): Root directory where the raw data are stored in
            BIDS format
        derivatives_path (Path): Root directory where the processed data are 
            stored. This directory contains the
            multimodal data to be used for the analysis.
        corr_output_path (Path): Directory where the correlations between 
            predicted and true brain states are stored
        features_filename (Optional[str | Path]): Name of the file containing 
            the features and their corresponding accuracies from 
            a previous run.
        overwrite (bool): Whether to overwrite existing files
        code_root (Path): Root directory for code
        tasks (list[str]): List of tasks to process
        subjects (list[str]): List of subjects to process
        sessions (list[str]): List of sessions to process
        runs (list[str]): List of runs to process
        additional_description (Optional[str]): Additional description in case 
            other specific methods, data, or subset of data are used.
    """
    n_threads: int = 32
    raw_path: Path = Path("/data2/Projects/eeg_fmri_natview/raw")
    derivatives_path: Path = Path("/data2/Projects/eeg_fmri_natview/derivatives")
    corr_output_path: Path = Path(f"data/eeg_bands_cpca/")
    features_filename: Optional[str | Path] = None 
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
    additional_description: Optional[str] = None

    def set_threads_nb(self, percentage: float = 50):
        """Set environment variables for thread control with validation

        Args:
            percentage (float): Percentage of the total number of 
                                threads to use

        Raises:
            ConfigurationError: If thread configuration is invalid
        """

        thread_vars = [
            "OMP_NUM_THREADS", 
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS"
        ]
        
        if percentage:
            self.n_threads = int(os.cpu_count() * (percentage / 100))

        for var in thread_vars:
            os.environ[var] = str(self.n_threads)
        
        return self

@dataclass
class MultimodalConfig:
    """Configuration class for multimodal data processing
    
    Attributes:
        resampling_factor (int): Factor by which to resample the data.
            This factor is the by how much the TR time is divided.
        sampling_rate_hz (float): Sampling rate in Hz.
        tr_time_seconds (float): TR time in seconds
        modalities (list[str]): List of modalities to include in the process.
        brainstates (BrainstatesConfig): Configuration instance 
            for the brainstates.
        eeg (EegConfig): Configuration instance for the EEG data
        eyetracking (EyeConfig): Configuration instance for the 
            eyetracking data.
    """
    resampling_factor: int = 8
    sampling_rate_hz: float = 3.8
    tr_time_seconds: float = 2.1
    modalities: List[str] = field(default_factory=lambda: ["brainstates", "eeg", "eyetracking"])
    brainstates: "BrainstatesConfig" = field(default_factory="BrainstatesConfig")
    eeg: "EegConfig" = field(default_factory="EegConfig")
    eyetracking: "EyeConfig" = field(default_factory="EyeConfig")

@dataclass
class EyeConfig:
    """Configuration class for eyetracking data processing
    
    Attributes:
        features (list[str]): List of features to include in the process.
            Usually the features are "pupil_dilation", "first_derivative",
            and "second_derivative".
        description (Optional[str]): The description regarding the 
            eyetracking data in the BIDS filename.
    """
    features: List[str] = field(default_factory=lambda: ["pupil_dilation"])
    description: Optional[str] = None

@dataclass
class BrainstatesConfig:
    """The configuration class for the Brainstate modality.
    
    Attributes:
        brainstates (list[str]): List of brainstates to include in the process.
            Names depends on the fMRI preprocessing and the methodology to 
            estimate brainstates.
        description (str): The description regarding the brainstate in the 
         BIDS filename.
    """
    brainstates: list[str] = field(default_factory=lambda: [
        'CAP1', 'CAP2', 'CAP3', 'CAP4', 'CAP5', 'CAP6', 'CAP7', 'CAP8'
    ])
    description: str = "Caps"

@dataclass
class EegConfig:
    """Configuration class for EEG data processing
    
    Attributes:
        description (str): Description of the EEG data
        sampling_rate_hz (float): The sampling rate of the EEG data in Hz.
        montage (str): EEG montage type.
        low_frequency_hz (float): Low frequency cutoff for high-pass filtering.
        high_frequency_hz (float): High frequency cutoff for low-pass 
            filtering.
        channels (Optional[list[str]]): List of channels to use.
        tmin (Optional[float]): Start time for analysis in seconds.
        tmax (Optional[float]): End time for analysis seconds.
    """
    description: str = "RawBk"
    sampling_rate_hz: float = 200
    montage: str = "easycap-M1"
    low_frequency_hz: float = 0.5
    high_frequency_hz: float = 40
    channels: Optional[List[str]] = None
    tmin: Optional[float] = None
    tmax: Optional[float] = None

@dataclass
class EegFeaturesConfig:
    """Configuration class for EEG feature extraction
    
    Attributes:
        frequencies (list[tuple[float, float]]): List of frequency bands to extract
    """
    frequencies: List[tuple[float, float]] = field(
        default_factory=lambda: [(0.5, 40)]
    )
    n_bands: int = len(frequencies)
    n_channels: int = 1

@dataclass
class ModelConfig:
    """Configuration class for model parameters
    
    This class is used when training and testing the Machine Learning model.
    
    Attributes:
        aggregation_function (Callable): In the case when feature selection is
            based on the best feature over the population, this attribute is
            used to calculate the metric for selecting the best feature on the
            entire population. Default is scipy.stats.ttest_samp.
        stat_func_kwargs: (Dict[str, Any]): This attribute is the key-words
            argument to feed to the `aggregation_function`. 
            Default is {"popmean": 0}.
        nb_desired_features (int): The number of features to get.
        estimator (sklearn.base.BaseEstimator): The scikit learn estimator
            to use for the model.
    """

    aggregation_function: Callable[[np.ndarray, float], tuple[float, float]] = scipy.stats.ttest_1samp
    stat_func_kwargs: Dict[str, Any] = field(default_factory=lambda: {"popmean": 0})
    nb_desired_features: int = 1

    estimator: sklearn.base.BaseEstimator = sklearn.linear_model.RidgeCV(cv=5)

    def select_features(self, 
                        eeg_channels: list[str | int],
                        eeg_bands: list[str | int],
                        eyetracking_features: Optional[list[str]] = None,
                        ) -> Dict[str, Any]:
        """Select the features to use for the model.
        
        Args:
            eyetracking_features (list[str]): The list of features to select
                for eyetracking data.
            eeg_channels (list[str | int]): The list of EEG channels to
                choose either by their name or by their index.
            eeg_bands (list[str | int]): The list of EEG frequency band to
                choose either by their name or by their index.

        """

        feature_set = {
            "eyetracking": 
                eyetracking_features,
            "eeg": {
                "channel": eeg_channels,
                "band": eeg_bands,
            }
        } 
        return feature_set
