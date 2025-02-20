"""Main script to train and test with feature selection (double dipping).
The aggregation across subject is done either with mean or median
"""

import os
from datetime import datetime

nthreads = "80" # 64 on synapse

os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
from pathlib import Path
import utils_sub_level_sessions as utils
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

def get_subjects_with_two_sessions(architecture: arch.BidsArchitecture) -> list[str]:
    subjects = architecture.database["subject"].unique()
    mask = (architecture.database.groupby("subject").session.nunique() == 2).values.tolist()
    subject_selected = subjects[np.where(mask)[0]]
    return list(subject_selected)

def main(config: "utils.ModelConfig") -> None:
    """Main function to orchestrate the feature selection process"""
    logger = utils.setup_logger(
        Path.home()/
        f"01_projects/eeg_brain_state_prediction/logs/"\
        f"double_dipping_{config.task}_WithPupil.log"
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
    subjects = get_subjects_with_two_sessions(architecture)
    logger.info(f"Found {len(architecture.subjects)} subjects to process")
    big_data = combine_data.pick_data(architecture=architecture)

         
    for subject in subjects:
        #if full_path.exists():
        #    logger.info(f"File already exists: {full_path}")
        #    continue
        logger.info(f"\nProcessing subject: {subject}")
        utils.pipeline(
            architecture=architecture, 
            subject=subject, 
            config=config,
            big_data=big_data,
        )
    logger.info("\nProcessing completed successfully")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature selection script")
    parser.add_argument("--task", type=str, default="checker", required=False, help="Task")
    parser.add_argument("--session", type=str, default = "02", required=False, help="Session")
    parser.add_argument("--additional_info", type=str, default = "EegOnly", required=False, help="Additional info")
    args = parser.parse_args()


    config = utils.ModelConfig(
        description = "CustomEnvBk8Caps",
        eeg_feature = "CustomEnv",
        caps = ["CAP1", 
                "CAP2", 
                "CAP3", 
                "CAP4", 
                "CAP5", 
                "CAP6", 
                "CAP7", 
                "CAP8"],
        nb_desired_features=range(1,51),
        data_directory="data/custom_envelope_caps/subject_level_feature_selection_session_level",
        task = args.task,
        additional_info=args.additional_info,
        session = args.session,
        n_threads = 32,
        features_data_filename="/home/slouviot/01_projects/eeg_brain_state_prediction/data/custom_envelope_caps/group_level/sub-all_task-checker_desc-CustomEnvBk_tstats.csv",
        
    )

    if args.additional_info == "EegOnly":
        config.feature_set = {
            "eyetracking": None,
            "eeg": {
                "channel": np.arange(config.n_channels).repeat(config.n_bands),
                "band": np.tile(np.arange(config.n_bands),config.n_channels),
            }
        }
    
    elif args.additional_info == "WithPupil":
        config.feature_set = {
            "eyetracking": ["pupil_dilation","first_derivative", "second_derivative"],
            "eeg": {
                "channel": np.arange(config.n_channels).repeat(config.n_bands),
                "band": np.tile(np.arange(config.n_bands),config.n_channels),
            }
        }
        
    main(config)
