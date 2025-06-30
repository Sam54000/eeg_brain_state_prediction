""" Functions to run the training/testing pipelines
"""
import os
from dotenv import load_dotenv

load_dotenv()

import scipy.stats
import sklearn
import pickle
import time
import numpy as np
from pathlib import Path
from itertools import product
import scipy
import logging
import sklearn.model_selection
import pandas as pd
import argparse
import combine_data as combine_data
import bids_explorer.architecture as arch
from typing import Dict, List, Callable, Optional, Union, Any, Tuple
from types import FunctionType
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    description: str 
    eeg_feature: str = "GfpBk"
    sampling_rate_hz: float = 3.8
    window_length_seconds: int = 10

    caps: np.ndarray = field(default_factory=lambda: np.array([
        'CAP1', 'CAP2', 'CAP3', 'CAP4', 'CAP5', 'CAP6', 'CAP7', 'CAP8'
    ]))

    n_bands: int = 1
    n_channels: int = 1

    aggregation_function: Callable[[np.ndarray, float], tuple[float, float]] = scipy.stats.ttest_1samp
    stat_func_kwargs: Dict[str, Any] = field(default_factory=lambda: {"popmean": 0})
    nb_desired_features: List[int] = field(
        default_factory=lambda: [ModelConfig.n_bands * ModelConfig.n_channels]
    )
    code_root: Path = Path(
        os.environ["HOME"],
        "01_projects",
        "eeg_brain_state_prediction",
    )
    data_root: Path = Path("/data2/Projects/eeg_fmri_natview/derivatives")
    runs: Optional[List[str]] = None
    task: str = "rest"
    additional_info: str = "All"
    feature_set = {
        "eyetracking": ["pupil_dilation", "first_derivative", "second_derivative"],
    }
    data_directory: str = f"data/eeg_bands_cpca/{eeg_feature}{additional_info}"
    n_threads: int = 32
    features_data_filename: Optional[str | Path] = None

def setup_logger(log_file=None):
    """Configure logging with timestamp and formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_bids_architecture(config: "ModelConfig") -> arch.BidsArchitecture:
    """Create BIDS architecture with given parameters"""
    parameters = {
        "root": config.data_root,
        "datatype": "multimodal",
        "suffix": "multimodal",
        "description": config.description,
        "run": "01",
        "task": config.task,
        "extension": ".pkl",
    }
    print(f"Creating BIDS architecture with parameters: {parameters}")
    print(f"Data root path exists: {config.data_root.exists()}")
    print(f"Full data root path: {config.data_root.absolute()}")
    
    architecture = arch.BidsArchitecture(**parameters)
    if hasattr(architecture, 'database'):
        print(f"Database info: {architecture.database.shape if hasattr(architecture.database, 'shape') else 'No shape attribute'}")
    return architecture

def train_model(X, Y) -> float:
    """Train model and return correlation coefficient"""
    estimator = sklearn.linear_model.RidgeCV(cv=5)
    estimator.fit(X, Y)
    return estimator

def process_single_iteration(big_data: Any, 
                             keys_list: list,
                             cap: str, 
                             feature_set: Dict, 
                             config: "ModelConfig"
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single iteration of data preparation and model training"""
    logger = logging.getLogger(__name__)
    logger.info(f"Create X and Y")
    X, Y = combine_data.create_X_and_Y(
        big_data = big_data,
        X_args = feature_set,
        cap_name = cap,
        window_length=int(
            config.sampling_rate_hz * config.window_length_seconds
            ),
        trim_args=(5, None),
        keys_list = keys_list,
    )

    logger.info(f"Build windowed mask")
    mask = combine_data.build_windowed_mask(
        big_data,
        key_list = keys_list,
        window_length=int(
            config.sampling_rate_hz * config.window_length_seconds
        ),
        trim_args = (5, None),
        features_args=feature_set
        )
    
    logger.info(f"Create X and Y with mask")
    X, Y = combine_data.arange_X_Y(
        X = X,
        Y = Y,
        mask = mask,
        group_rejection = False
    )

    logger.info(f"Train model")
    estimator = train_model(X, Y)

    return estimator


def save_model(model: Any,
               path: Path) -> None:
    """Save model"""
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def make_saving_path(config: "ModelConfig", 
                     task: str,
                     cap_name: str,
                     model_name: str) -> Path:
    """Make saving path"""
    output_path = config.code_root / config.data_directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"sub-all_task-{task}_desc-{model_name}{cap_name}_model.pkl"
    full_path = output_path / filename
    return full_path

def pipeline(
    architecture: 'arch.BidsArchitecture', 
    config: "ModelConfig",
) -> None:
    """Run iterative feature selection process.
    
    Args:
        architecture: BIDS architecture object
        subject: Subject identifier
        config: Model configuration
        aggregated_selection: DataFrame with feature selection results
        task: Task identifier
        description: Description string
        aggregation_mode: Method for aggregating results
    """
    logger = logging.getLogger(__name__)
    big_data = combine_data.pick_data(architecture=architecture)
    for task in config.task:
        selection = architecture.select(task=task)
        for cap in config.caps:
            try:
                logger.info(f"Processing task: {task}, cap: {cap}")
                estimator = process_single_iteration(
                    big_data = big_data,
                    keys_list = selection.database.index.values,
                    cap = cap,
                    feature_set = config.feature_set,
                    config = config
                )
                path = make_saving_path(config, task, cap, "ridge")
                logger.info(f"Save model to: {path}")
                save_model(model = estimator,
                           path = path)
                
                    
            except Exception as e:
                logger.error(f"Error processing feature set: {config.feature_set}, cap: {cap}")
                logger.error(f"Exception details: {str(e)}", exc_info=True)
                raise e

        logger.info(f"Saving results to: {make_saving_path(config, task, cap, "model")}")