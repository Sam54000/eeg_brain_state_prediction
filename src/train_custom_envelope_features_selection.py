"""Main script to train and test with feature selection (double dipping).
The aggregation across subject is done either with mean or median
"""

import os
nthreads = "32" # 64 on synapse

os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import scipy.stats
import sklearn
import time
import numpy as np
from pathlib import Path
import scipy
import sklearn.model_selection
import pandas as pd
import argparse
import src.training_testing_pipeline.combine_data as combine_data
import bids_explorer.architecture as arch
from typing import Dict, List, Callable, Optional, Union, Any, Tuple
from types import FunctionType
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    sampling_rate_hz: float = 3.8
    window_length_seconds: int = 10
    caps: np.ndarray = field(default_factory=lambda: np.array([
        'CAP1', 'CAP2', 'CAP3', 'CAP4', 'CAP5', 'CAP6', 'CAP7', 'CAP8'
    ]))
    n_bands: int = 39
    n_channels: int = 61
    aggregation_function: Callable[[np.ndarray, float], tuple[float, float]] = scipy.stats.ttest_1samp
    stat_func_kwargs: Dict[str, Any] = field(default_factory=lambda: {"popmean": 0})
    nb_desired_features: List[int] = field(
        default_factory=lambda: np.arange(1, 51, 1)
    )

def set_thread_env(nthreads: str = "32") -> None:
    """Set environment variables for thread control"""
    thread_vars = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
    ]
    for var in thread_vars:
        os.environ[var] = nthreads

def create_bids_architecture(root_path: str, task: str) -> arch.BidsArchitecture:
    """Create BIDS architecture with given parameters"""
    parameters = {
        "root": Path(root_path),
        "datatype": "multimodal",
        "suffix": "multimodal",
        "description": "CustomEnvBk8Caps",
        "run": "01",
        "task": task,
        "extension": ".pkl",
    }
    return arch.BidsArchitecture(**parameters)

def initialize_results_dict() -> Dict:
    """Initialize the dictionary for storing results"""
    return {
        'subject': [],
        'session': [],
        'ts_CAPS': [],
        'pearson_r': [],
        'frequency_Hz': [],
        'electrode': [],
        'n_features': []  # New column for tracking feature count
    }

def train_and_evaluate_model(X_train: np.ndarray, Y_train: np.ndarray, 
                           X_test: np.ndarray, Y_test: np.ndarray) -> float:
    """Train model and return correlation coefficient"""
    estimator = sklearn.linear_model.RidgeCV(cv=5)
    estimator.fit(X_train, Y_train)
    Y_hat = estimator.predict(X_test)
    return np.corrcoef(Y_test.T, Y_hat.T)[0, 1]

def process_single_iteration(big_data: Any, train_keys: List, test_keys: List,
                           cap: str, feature_set: Dict, config: ModelConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def save_results(results_df: pd.DataFrame, 
                 subject: str, 
                 task: str, 
                 description: str,
                 aggregation_mode_str: str,
                 n_features: int,
                 ) -> None:
    """Save results to CSV file"""
    output_path = "/home/slouviot/01_projects/eeg_brain_state_prediction/data/"\
        f"custom_envelope/group_level_feature_selection/sub-{subject}_task-{task}_desc-{description}FeatureSelectionAgg"\
        f"{aggregation_mode_str.capitalize()}WithPupil_predictions.csv"
    results_df.to_csv(output_path, index=False)

def get_all_features_dataframe(csv_file: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_file)

def aggregate_df_across_subjects(dataframe: pd.DataFrame,
                                 config: ModelConfig,
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
    to_sort: str = "pearson_r",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate feature combinations based on number of features.
    
    Args:
        n_features: Number of features to select
        aggregated_selection: DataFrame containing feature selection data
        subject: Optional subject information
        config: Model configuration
        tstats (bool): Wether the dataframe provided is the tstats or not.
        
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

def pipeline(
    architecture: 'arch.BidsArchitecture',  # Using string literal for forward reference
    subject: str, 
    config: ModelConfig,
    aggregated_selection: pd.DataFrame,
    task: str = 'rest',
    description: str = 'CustomEnvBkCaps',
    aggregation_mode: Union[str, FunctionType] = "median",
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
    aggregation_mode_str = (
        aggregation_mode.__name__ 
        if isinstance(aggregation_mode, FunctionType)
        else aggregation_mode
    )
    
    """Run iterative feature selection process"""
    big_data = combine_data.pick_data(architecture=architecture)
    train_arch, test_arch = combine_data.generate_train_test_architectures(
        architecture=architecture,
        train_subjects=architecture.subjects,
        test_subjects=subject
    )

    results = initialize_results_dict()
    for n_features in config.nb_desired_features:#range(max_features - 1, max_features, nb_feat_steps):
        
        for test_keys, test_session in test_arch:
            train_keys = train_arch.database.index.values
            
            for cap in config.caps:
                try:
                    feature_set = get_best_n_feature_combinations(
                        n_features,
                        aggregated_selection[aggregated_selection['ts_CAPS'] == cap],
                        to_sort="t_stat",
                    )
                    #feature_set['eyetracking'] = ["pupil_dilation", "first_derivative", "second_derivative"]
                    X_train, Y_train, X_test, Y_test = process_single_iteration(
                        big_data, 
                        train_keys,
                        [test_keys],
                        cap,
                        feature_set, 
                        config
                    )
                    
                    if any(shape == 0 for shape in [*X_train.shape, *Y_train.shape, 
                                                    *X_test.shape, *Y_test.shape]):
                        continue
                        
                    r = train_and_evaluate_model(X_train, Y_train, X_test, Y_test)
                    
                    # Update results
                    results['subject'].append(subject)
                    results['session'].append(test_session['session'])
                    results['ts_CAPS'].append(cap)
                    results['pearson_r'].append(r)
                    results['frequency_Hz'].append([
                        f + 1 for f in feature_set['eeg']['band']
                        ])
                    results['electrode'].append(feature_set['eeg']['channel'])
                    results['n_features'].append(n_features)
                    
                except Exception as e:
                    print(f"Error processing feature set: whatever, cap: {cap}")
                    raise e
                    print(e)
                    continue
        
    results_df = pd.DataFrame(results)
    save_results(results_df, 
                 subject, 
                 task, 
                 description, 
                 aggregation_mode_str, 
                 n_features)

def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate the feature selection process"""
    set_thread_env()
    config = ModelConfig()
    architecture = create_bids_architecture("/data2/Projects/eeg_fmri_natview/derivatives", args.task)
    #features_dataframe = get_all_features_dataframe(
    #    "/home/slouviot/01_projects/eeg_brain_state_prediction/data/"\
    #    "custom_envelope_caps/group_level/"\
    #    f"sub-all_task-{args.task}_desc-{args.desc}_predictions.csv"
    #)
    #     
    #aggregated_selection = aggregate_df_across_subjects(
    #    features_dataframe,
    #    config=config,
    #    )
    aggregated_selection = pd.read_csv(
        "/home/slouviot/01_projects/eeg_brain_state_prediction/data/"\
        "custom_envelope_caps/group_level/"\
        f"sub-all_task-{args.task}_desc-{args.desc}_tstats.csv"
    )
        

    for subject in architecture.subjects:
        pipeline(
            architecture=architecture, 
            subject=subject, 
            config=config,
            aggregated_selection=aggregated_selection,
            task = args.task,
            description=args.desc,
            aggregation_mode='tstats',
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_custom_envelope_features_selection',
        description='Train and test with feature selection across subjects'
    )
    parser.add_argument('--subject', default='01', help='Subject identifier')
    parser.add_argument('--task', default="rest", help='Task identifier')
    parser.add_argument(
        '--agg',
        default="ttstats",
        help='How to combine the correlation across subjects: median or mean'
    )
    parser.add_argument('--desc', default='CustomEnvBk', help='Description string')
    
    args = parser.parse_args()
    main(args)
