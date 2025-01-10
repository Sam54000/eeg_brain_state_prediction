"""Main script to train and test with feature selection (double dipping).
The aggregation across subject is done either with mean or median
"""

import os
import sklearn
import numpy as np
from pathlib import Path
import sklearn.model_selection
import pandas as pd
import argparse
import combine_data
import bids_explorer.architecture as arch
from typing import Dict, List, Tuple, Any
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

def set_thread_env(nthreads: str = "1") -> None:
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
        "description": "CustomEnvBk8",
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
                 aggregation_mode: str,
                 ) -> None:
    """Save results to CSV file"""
    output_path = "/home/slouviot/01_projects/eeg_brain_state_prediction/data"\
        f"/sub-{subject}_task-{task}_desc-{description}FeatureSelectionAgg"\
        f"{aggregation_mode.capitalize()}_predictions.csv"
    results_df.to_csv(output_path, index=False)

def get_all_features_dataframe(csv_file: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_file)

def aggregate_df_across_subjects(dataframe: pd.DataFrame,
                                 stat_func: str = 'mean') -> pd.DataFrame:
    "When getting features for the entire population."
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

    aggregated = getattr(grouped,stat_func)()
    return aggregated.reset_index()

    
def get_best_n_feature_combinations(
    n_features: int, 
    cap: str,
    features_dataframe: pd.DataFrame,
    subject: dict | None,
    aggregation_mode: str = "median",
    ) -> Dict[str,Dict[str,List]]: # TODO: TO MODIFY
    """Generate feature combinations based on number of features"""

    if subject is not None:
        boolean_indices = (
            (features_dataframe["subject"] == subject) & 
            (features_dataframe["ts_CAPS"] == cap)
        )
    else:
        features_dataframe = aggregate_df_across_subjects(
            features_dataframe, 
            stat_func=aggregation_mode,
            )
        boolean_indices = features_dataframe["ts_CAPS"] == cap
            
    features_dataframe = features_dataframe.loc[boolean_indices]
    features_dataframe.sort_values(by='pearson_r', 
                                   ascending=False,
                                   inplace = True)
    channels = features_dataframe['electrode'].values[:n_features]
    bands = features_dataframe['frequency_Hz'].apply(lambda x: x-1).values[:n_features]
    feature_sets = {
        "eeg": {
            "channel": channels,
            "band": bands,
        }
    }
    
    return feature_sets

def pipeline(
    architecture: arch.BidsArchitecture, 
    subject: str, 
    config: ModelConfig,
    max_features:int,
    features_dataframe: pd.DataFrame,
    nb_feat_steps: int = 4,
    task: str = 'rest',
    description: str = 'CustomEnvBk',
    aggregation_mode: str = "median",

    ) -> None:
    """Run iterative feature selection process"""
    big_data = combine_data.pick_data(architecture=architecture)
    train_arch, test_arch = combine_data.generate_train_test_architectures(
        architecture=architecture,
        train_subjects=architecture.subjects,
        test_subjects=subject
    )

    results = initialize_results_dict()
    for n_features in range(1, max_features + 1, nb_feat_steps):
        
        for test_keys, test_session in test_arch:
            train_keys = train_arch.database.index.values
            
            for cap in config.caps:
                try:
                    feature_set = get_best_n_feature_combinations(
                        n_features,
                        features_dataframe=features_dataframe,
                        cap = cap,
                        subject = None,
                        aggregation_mode=aggregation_mode
                    )
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
                    print(f"Error processing feature set: {feature_set}, cap: {cap}")
                    print(e)
                    continue
        
    results_df = pd.DataFrame(results)
    save_results(results_df, subject, task, description, aggregation_mode)

def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate the feature selection process"""
    set_thread_env()
    config = ModelConfig()
    architecture = create_bids_architecture("/data2/Projects/eeg_fmri_natview/derivatives", args.task)
    features_dataframe = get_all_features_dataframe(
        "/home/slouviot/01_projects/eeg_brain_state_prediction/data/"\
        f"sub-all_task-{args.task}_desc-{args.desc}_predictions.csv"
    )
         
    pipeline(
        architecture=architecture, 
        subject=args.subject, 
        config=config,
        max_features=len(features_dataframe),
        features_dataframe=features_dataframe,
        task = args.task,
        description=args.desc,
        aggregation_mode=args.agg,
        nb_feat_steps=1,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train_custom_envelope_features_selection')
    parser.add_argument('--subject', default='01')
    parser.add_argument('--task', default="checker")
    parser.add_argument(
        '--agg',
        default="median",
        help='How to combine the correlation across subjects: median or mean'
    )
    parser.add_argument('--desc', default = 'CustomEnvBk')
    args = parser.parse_args()
    main(args)