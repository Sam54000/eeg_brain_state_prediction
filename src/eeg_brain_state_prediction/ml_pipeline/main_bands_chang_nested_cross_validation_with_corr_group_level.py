"""Main script to train and test with feature selection (double dipping).
The aggregation across subject is done either with mean or median
"""

import os
from datetime import datetime

nthreads = "110" # 64 on synapse

os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
from pathlib import Path
import utils_bands_chang_nested_cross_validation_with_corr_group_level as utils
import cross_correlation
import numpy as np
import bids_explorer.architecture as arch
import combine_data
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Feature selection script")
    parser.add_argument("--task", type=str, default="checker", required=True, help="Task")
    parser.add_argument("--session", type=str, default = "02", required=True, help="Session")
    parser.add_argument("--additional_info", type=str, default = "WithPupil", required=True, help="Additional info")
    return parser.parse_args()

def main(config: "utils.ModelConfig") -> None:
    """Main function to orchestrate the feature selection process"""
    logger = utils.setup_logger(
        Path.home()/
        f"01_projects/eeg_brain_state_prediction/logs/"\
        f"nested_cross_val_chang_{config.task}.log"
    )
    
    logger.info(f"Starting processing with task: {config.task}, description: {config.description}")
    
    # Debug environment
    logger.debug("\nEnvironment Debug:")
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"HOME environment: {os.environ.get('HOME')}")
    logger.debug(f"Python path: {os.environ.get('PYTHONPATH')}")
    
    config.runs = None
    
    logger.info(f"\nConfig Debug:")
    logger.info(f"Code root: {config.code_root}")
    logger.info(f"Data root: {config.data_root}")
    logger.info(f"Data directory: {config.data_directory}")

    architecture = utils.create_bids_architecture(config)
    logger.info(f"Found {len(architecture.subjects)} subjects to process")
    big_data = combine_data.pick_data(architecture=architecture)
    max_xcorr, info = cross_correlation.population_corr(architecture,big_data)
    stats = cross_correlation.calculate_sorted_tstat(
        architecture,
        max_xcorr,
        info
        )

         
    for subject in architecture.subjects:
        logger.info(f"\nProcessing subject: {subject}")
        utils.pipeline(
            architecture=architecture, 
            subject=subject, 
            config=config,
            big_data=big_data,
            stats = stats,
        )
    logger.info("\nProcessing completed successfully")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature selection script")
    parser.add_argument("--task", type=str, default="MeRest", required=False, help="Task")
    parser.add_argument("--session", type=str, default = "01", required=False, help="Session")
    parser.add_argument("--additional_info", type=str, default = "GroupLevel", required=False, help="Additional info")
    args = parser.parse_args()


    config = utils.ModelConfig(
        description = "SSDbandsEnv",
        eeg_feature = "SSDbandsEnv",
        caps = ['CAP1',
                'CAP2',
                'CAP3',
                'CAP4',
                'CAP5',
                'CAP6',
                'CAP7',
                'CAP8',
                'GS'],
        nb_desired_features=30,
        data_root=Path("/data2/Projects/eeg_fmri_natview/chang_data/derivatives"),
        data_directory="/home/slouviot/01_projects/eeg_brain_state_prediction/data/chang_data/eeg_bands",
        task = args.task,
        additional_info=args.additional_info,
        n_threads = 32,
        features_data_filename="/home/slouviot/01_projects/eeg_brain_state_prediction/data/custom_envelope_caps/group_level/sub-all_task-checker_desc-CustomEnvBk_tstats.csv",
        
    )

    config.feature_set = {
        "eeg": {
            "channel": 0,
            "band": 0,
        }
    }
    
    main(config)
