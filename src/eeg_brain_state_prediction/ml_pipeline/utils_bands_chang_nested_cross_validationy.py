""" Functions to run the training/testing pipelines
"""
import os
from dotenv import load_dotenv

load_dotenv()

import scipy.stats
import sklearn
import time
import numpy as np
from pathlib import Path
from itertools import product
import scipy
import logging
import sklearn.model_selection
from sklearn.model_selection import train_test_split
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
        "eyetracking": None,
        "eeg": {
            "channel": np.arange(n_channels).repeat(n_bands),
            "band": np.tile(np.arange(n_bands),n_channels),
        }
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

def initialize_results_dict() -> Dict:
    """Initialize the dictionary for storing results"""
    return {
        "subject": [],
        "session": [],
        "ts_CAPS": [],
        "pearson_r": [],
        "frequency_Hz": [],
        "electrode": [],
        "n_features": [],
    }

def train_and_evaluate_model(X_train: np.ndarray, Y_train: np.ndarray, 
                           X_test: np.ndarray, Y_test: np.ndarray) -> float:
    """Train model and return correlation coefficient"""
    estimator = sklearn.linear_model.RidgeCV(cv=5)
    estimator.fit(X_train, Y_train)
    Y_hat = estimator.predict(X_test)
    return np.corrcoef(Y_test.T, Y_hat.T)[0, 1]

def process_single_iteration(big_data: Any, 
                             train_keys: List, 
                             test_keys: List,
                             cap: str, 
                             feature_set: Dict, 
                             config: "ModelConfig"
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single iteration of data preparation and model training"""
    return combine_data.create_train_test_data(
        big_data=big_data,
        train_keys=train_keys,
        test_keys=test_keys,
        cap_name=cap,
        features_args=feature_set,
        window_length=int(config.sampling_rate_hz * config.window_length_seconds),
        masking=True,
        trim_args=(5, -5)
    )

def make_saving_path(config: "ModelConfig", subject: str) -> Path:
    """Make saving path"""
    output_path = config.code_root / config.data_directory
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"sub-{subject}_task-{config.task}_desc-{config.description}{config.additional_info}_predictions.csv"
    full_path = output_path / filename
    return full_path

def save_results(results_df: pd.DataFrame, 
                 full_path: Path,
                 ) -> None:
    print(f"Saving results to: {full_path}")
    results_df.to_csv(full_path, index=False)
    print(f"File saved successfully with {len(results_df)} rows")

def nested_cross_val(train_arch: arch.BidsArchitecture,
                     iteration: int,
                     config: "ModelConfig",
                     big_data:dict,
                     cap: str,
                     ) -> None:
    """Run nested cross validation"""
    results = {
        "iteration": [],
        "pearson_r": [],
        "frequency_Hz": [],
        "electrode": [],
    }
    channels = [ch for ch in range(29)]
    bands = [band for band in range(5)]
    combination = product(channels, bands)
    for chan, band in combination:
        print(
            f"Channel: {chan}\nBand: {band}"
        )
        for i in range(iteration):
            train_subjects, test_subjects = train_test_split(train_arch.subjects)
            nested_train_arch = train_arch.select(subject = train_subjects)
            nested_test_arch = train_arch.select(subject = test_subjects)
            X_train, Y_train, X_test, Y_test = process_single_iteration(
                cap = cap,
                big_data=big_data,
                train_keys=nested_train_arch.database.index,
                test_keys=nested_test_arch.database.index,
                feature_set={
                    "eeg":{
                        "channel":chan,
                        "band":band,
                    }
                },
                config=config,
                )
            r = train_and_evaluate_model(X_train,
                                                    Y_train,
                                                    X_test,
                                                    Y_test)
            results["iteration"].append(i)
            results["pearson_r"].append(r)
            results["frequency_Hz"].append(band)
            results["electrode"].append(chan)

    df = pd.DataFrame(results)
    # Calculate t-test for each frequency/electrode combination
    ttest_results = []
    for freq in df['frequency_Hz'].unique():
        for elec in df['electrode'].unique():
            mask = (df['frequency_Hz'] == freq) & (df['electrode'] == elec)
            r_values = df[mask]['pearson_r']
            if len(r_values) > 0:
                t_stat, p_val = scipy.stats.ttest_1samp(r_values, 0)
                ttest_results.append({
                    'frequency_Hz': freq,
                    'electrode': elec, 
                    't_stat': t_stat,
                    'p_value': p_val
                })
    
    ttest_df = pd.DataFrame(ttest_results)
    ttest_df.sort_values(by="t_stat", ascending=False, inplace=True)
    return ttest_df
    

def pipeline(
    architecture: 'arch.BidsArchitecture', 
    subject: str, 
    config: "ModelConfig",
    big_data:dict,
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
    
    train_arch, test_arch = combine_data.generate_train_test_architectures(
        architecture=architecture,
        train_subjects=architecture.subjects,
        test_subjects=subject
    )

    results = initialize_results_dict()
        
    for cap in config.caps:
        for test_keys, test_session in test_arch:
            train_keys = train_arch.database.index.values
        
            best_features = nested_cross_val(
                cap = cap,
                train_arch=train_arch,
                iteration = 5,
                config=config,
                big_data=big_data
            )
            for n_features in range(1,config.nb_desired_features+1):
                print(f"Processing {n_features} features")
                config.feature_set.update(
                    {"eeg":{
                        "channel":best_features["electrode"].values[:n_features],
                        "band": best_features["frequency_Hz"].values[:n_features],
                    }
                    }
                )
                try:
                    X_train, Y_train, X_test, Y_test = process_single_iteration(
                        big_data, 
                        train_keys,
                        [test_keys],
                        cap,
                        config.feature_set, 
                        config
                    )
                    
                    if any(shape == 0 for shape in [*X_train.shape, *Y_train.shape, 
                                                    *X_test.shape, *Y_test.shape]):
                        continue
                        
                    r = train_and_evaluate_model(X_train, Y_train, X_test, Y_test)
                    
                    results['subject'].append(subject)
                    results['session'].append(test_session['session'])
                    results['ts_CAPS'].append(cap)
                    results['pearson_r'].append(r)
                    eeg_spec = config.feature_set.get('eeg')
                    if eeg_spec is not None:
                        results['frequency_Hz'].append(eeg_spec.get('band'))
                        results['electrode'].append(eeg_spec.get('channel'))
                    else:
                        results['frequency_Hz'].append(None)
                        results['electrode'].append(None)
                    results['n_features'].append(n_features)
                    
                except Exception as e:
                    logger.error(f"Error processing feature set: {config.feature_set}, cap: {cap}")
                    logger.error(f"Exception details: {str(e)}", exc_info=True)
                    raise e

    results_df = pd.DataFrame(results)
    results_df["task"] = config.task
    results_df["description"] = config.description
    logger.info(f"Saving results to: {make_saving_path(config, subject)}")
    save_results(results_df = results_df, 
                 full_path = make_saving_path(config, subject)) 