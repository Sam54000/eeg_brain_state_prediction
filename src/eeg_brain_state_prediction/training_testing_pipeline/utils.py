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
        "eyetracking": 
            ["pupil_dilation", "first_derivative", "second_derivative"],
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

def set_thread_env(config) -> None:
    """Set environment variables for thread control"""
    thread_vars = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
    ]
    for var in thread_vars:
        os.environ[var] = str(config.n_threads)

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
        trim_args=(5, None)
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

def get_all_features_dataframe(csv_file: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_file)

def aggregate_df_across_subjects(dataframe: pd.DataFrame,
                                 config: "ModelConfig",
                                 ) -> pd.DataFrame:
    """When getting features for the entire population."""
    if isinstance(config.aggregation_function, str):
        func = config.aggregation_function
    else:
        def func(x):
            return scipy.stats.ttest_1samp(x, popmean=0).statistic

    grouped = dataframe[[
        "frequency_Hz",
        "electrode",
        "ts_CAPS",
        "pearson_r"
    ]].groupby([
        "frequency_Hz",
        "electrode",
        "ts_CAPS",
    ])

    aggregated = grouped.aggregate(func)
    return aggregated.reset_index()

def get_best_n_feature_combinations(
    n_features: int, 
    aggregated_selection: pd.DataFrame,
    to_sort: str = "t_stat",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate feature combinations based on number of features.
    
    Args:
        n_features: Number of features to select
        aggregated_selection: DataFrame containing feature selection data
        subject: Optional subject information
        config: Model configuration
        
    Returns:
        Dictionary containing selected channels and bands
    """
    aggregated_selection = aggregated_selection.sort_values(
        by=to_sort, 
        ascending=False,
    )

    channels = aggregated_selection['electrode'].values[:n_features]
    bands = aggregated_selection['frequency_Hz'].apply(lambda x: x-1).values[:n_features]
    
    return {
        "eeg": {
            "channel": channels,
            "band": bands,
        },
        "eyetracking": ["pupil_dilation", 
                        "first_derivative",
                        "second_derivative"]
    }

def pipeline(
    architecture: 'arch.BidsArchitecture', 
    subject: str, 
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
    train_arch, test_arch = combine_data.generate_train_test_architectures(
        architecture=architecture,
        train_subjects=architecture.subjects,
        test_subjects=subject
    )

    results = initialize_results_dict()
    if config.features_data_filename is not None:
        features_csv = pd.read_csv(config.features_data_filename)
    else:
        config.nb_desired_features = [1]
    for n_features in config.nb_desired_features:
        
        for test_keys, test_session in test_arch:
            train_keys = train_arch.database.index.values
            
            for cap in config.caps:
                try:
                    config.feature_set = get_best_n_feature_combinations(
                        n_features=n_features,
                        aggregated_selection=features_csv[features_csv["ts_CAPS"] == cap],
                        to_sort = "t_stat",
                    )
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