""" Functions to run the training/testing pipelines
"""
import os
from dotenv import load_dotenv
from eeg_brain_state_prediction.logger import setup_logger, log_execution

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
import combine_data as combine_data
import bids_explorer.architecture as arch
from typing import Dict, List, Callable, Optional, Union, Any, Tuple
from types import FunctionType
import eeg_brain_state_prediction.tools.configs as configs
from dataclasses import dataclass, field

def setup_logger(log_file=None):
    """Wrapper around the centralized logger setup for backward compatibility"""
    return setup_logger(__name__, log_file)

@log_execution()
def create_bids_architecture(
    config: configs.PipelineConfig
    ) -> arch.BidsArchitecture:
    """Create BIDS architecture with given parameters.

    Create the BidsArchitecture for accessing to the multimodal data in 
    an ordered manner.

    Args: 
        config (configs.PipelineConfig): The configuration instance dedicated
            to the pipeline.
    
    Returns:
        arch.BidsArchitecture: The architecture instance.
    
    """
    logger = logging.getLogger(__name__)
    parameters = {
        "root": config.data_root,
        "datatype": "multimodal",
        "suffix": "multimodal",
        "description": config.description,
        "run": "01",
        "task": config.task,
        "extension": ".pkl",
    }
    logger.info(f"Creating BIDS architecture with parameters: {parameters}")
    logger.info(f"Data root path exists: {config.data_root.exists()}")
    logger.info(f"Full data root path: {config.data_root.absolute()}")
    
    architecture = arch.BidsArchitecture(**parameters)
    if hasattr(architecture, 'database'):
        logger.info(f"Database info: {architecture.database.shape if hasattr(architecture.database, 'shape') else 'No shape attribute'}")
    return architecture

def initialize_results_dict() -> Dict:
    """Initialize the dictionary for storing results.
    
    Returns:
        dict: The dictionary to populate in the loop and transform in a 
        `pandas.DataFrame` at the end of the process.
    """

    return {
        "subject": [],
        "session": [],
        "ts_CAPS": [],
        "pearson_r": [],
        "frequency_Hz": [],
        "electrode": [],
        "n_features": [],
    }

def train_and_evaluate_model(
    estimator: sklearn.base.BaseEstimator,
    X_train: np.ndarray, 
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray
    ) -> float:
    """Train model and return correlation coefficient.
    
    This function train the model by fitting the desired estimator.

    Args:
        estimator (sklearn.bas.BasEstimator): The estimator to use to fit the 
            data.
        X_train (np.ndarray): The training data.
        Y_train (np.ndarray): The training targets.
        X_test (np.ndarray): The test data.
        Y_test (np.ndarray): The real test target to compare to the estimated
            ones.
    
    Returns:
        float: The Pearson' correlation between predicated and real values.
    
    """
    estimator.fit(X_train, Y_train)
    Y_hat = estimator.predict(X_test)
    return np.corrcoef(Y_test.T, Y_hat.T)[0, 1]

def transform(big_data: Dict[int, Dict[Any]], 
              train_keys: List[int], 
              test_keys: List,
              brainstate: str, 
              feature_set: Dict, 
              config: configs.PipelineConfig
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform the multimodal data into format to use with sklearn.

    Transform the multimodal data into predictors, target, and train
    and test sets.
    
    Args:
        big_data 
    
    """
    X_train, Y_train, X_test, Y_test = combine_data.create_train_test_data(
        big_data=big_data,
        train_keys=train_keys,
        test_keys=test_keys,
        brainstate=brainstate,
        features_args=feature_set,
        window_length=int(config.sampling_rate_hz*config.window_length_seconds),
        masking=True,
        trim_args=(5, None)
    )
    return X_train, Y_train, X_test, Y_test

def make_saving_path(config: configs.PipelineConfig, subject: str) -> Path:
    """Make saving path"""
    output_path = config.code_root / config.data_directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"sub-{subject}_task-{config.task}_desc-{config.description}{config.additional_info}_predictions.csv"
    full_path = output_path / filename
    return full_path

@log_execution()
def save_results(results_df: pd.DataFrame, 
                 full_path: Path,
                 ) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Saving results to: {full_path}")
    results_df.to_csv(full_path, index=False)
    logger.info(f"File saved successfully with {len(results_df)} rows")

def get_all_features_dataframe(csv_file: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_file)

def aggregate_df_across_subjects(dataframe: pd.DataFrame,
                                 config: configs.ModelConfig,
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
        }
    }

for subject in ["01"]:#architecture.subjects:
    full_path = utils.make_saving_path(config, subject)
    #if full_path.exists():
    #    logger.info(f"File already exists: {full_path}")
    #    continue
    logger.info(f"\nProcessing subject: {subject}")
    utils.pipeline(
        architecture=architecture, 
        subject=subject, 
        config=config,
    )

@log_execution()
def pipeline_1(
    architecture: 'arch.BidsArchitecture', 
    subject: str, 
    config: 
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
        logger.info(f"Processing {n_features} features")
        
        for test_keys, test_session in test_arch:
            train_keys = train_arch.database.index.values
            
            for cap in config.caps:
                try:
                    config.feature_set.update(get_best_n_feature_combinations(
                        n_features=n_features,
                        aggregated_selection=features_csv[features_csv["ts_CAPS"] == cap],
                        to_sort = "t_stat",
                    ))
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

def best_features_screening(pipeline_config: configs.PipelineConfig,
             multimodal_config: configs.MultimodalConfig,
             model_config: configs.ModelConfig):
    for brainstate in multimodal_config.brainstates.brainstates:
        best_features_combinations = get_best_n_feature_combinations(
            n_features=n_features,
            aggregated_selection=features_csv[features_csv["brainstate"] == brainstate],
            to_sort = "t_stat",
        )
        atomic_pipelines()

#Maybe make different kind of piplines as decorators because they are wrapping
#The atomic pipeline function

def atomic_pipelines(big_data: dict,
                     train_keys: list,
                     test_keys: list,
                     feature_set: dict,
                     brainstate: str,
                     pipeline_config: configs.PipelineConfig,
                     model_config: configs.ModelConfig) -> dict:

    if not isinstance(test_keys, list):
        test_keys = list(test_keys)

    sampling_rate, window_length_seocnds = (
        pipeline_config.sampling_rate_hz,
        pipeline_config.window_length_seconds
    )

    X_train, Y_train, X_test, Y_test = combine_data.create_train_test_data(
        big_data=big_data,
        train_keys=train_keys,
        test_keys=test_keys,
        brainstate=brainstate,
        features_args=feature_set,
        window_length=int(sampling_rate*window_length_seocnds),
        masking=True,
        trim_args=(5, None)
    )
    
    sets_shape = [*X_train.shape,
                  *Y_train.shape,
                  *X_test.shape,
                  *Y_test.shape]
    

    if any(shape == 0 for shape in  sets_shape):
        return

    r = train_and_evaluate_model(
        model_config = model_config,
        X_train      = X_train, 
        Y_train      = Y_train, 
        X_test       = X_test, 
        Y_test       = Y_test,
        )

    
    results['subject'].append(subject)
    results['session'].append(test_session['session'])
    results[brainstate].append(brainstate)
    results['pearson_r'].append(r)
    eeg_spec = config.feature_set.get('eeg')
    if eeg_spec is not None:
        results['frequency_Hz'].append(eeg_spec.get('band'))
        results['electrode'].append(eeg_spec.get('channel'))
    else:
        results['frequency_Hz'].append(None)
        results['electrode'].append(None)

    results['n_features'].append(n_features)
    
