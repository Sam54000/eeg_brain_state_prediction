"""Main script to train and test with feature selection (double dipping).
The aggregation across subject is done either with mean or median
"""

import os
from datetime import datetime

nthreads = "32" # 64 on synapse

os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
from pathlib import Path
import utils
import numpy as np
import bids_explorer.architecture as arch


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
    
    utils.set_thread_env(config)
    config.runs = None
    
    logger.info(f"\nConfig Debug:")
    logger.info(f"Code root: {config.code_root}")
    logger.info(f"Data root: {config.data_root}")
    logger.info(f"Data directory: {config.data_directory}")

    architecture = utils.create_bids_architecture(config)
    logger.info(f"Found {len(architecture.subjects)} subjects to process")
         
    for subject in architecture.subjects:
        full_path = utils.make_saving_path(config, subject)
        if full_path.exists():
            logger.info(f"File already exists: {full_path}")
            continue
        logger.info(f"\nProcessing subject: {subject}")
        utils.pipeline(
            architecture=architecture, 
            subject=subject, 
            config=config,
        )
    logger.info("\nProcessing completed successfully")

if __name__ == "__main__":

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
        data_directory="data/custom_envelope_caps/group_level_feature_selection",
        task = "rest",
        additional_info="WithPupil",
        n_threads = 32,
        features_data_filename="/home/slouviot/01_projects/eeg_brain_state_prediction/data/custom_envelope_caps/group_level/sub-all_task-rest_desc-CustomEnvBk_tstats.csv",
        
    )
        
    utils.set_thread_env(config)
        
    main(config)
