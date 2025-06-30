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
import utils_pupil_training as utils
import numpy as np
import bids_explorer.architecture as arch


def main(config: "utils.ModelConfig") -> None:
    """Main function to orchestrate the feature selection process"""
    logger = utils.setup_logger(
        Path.home()/
        f"01_projects/eeg_brain_state_prediction/logs/"\
        f"pupil_training_{config.task}.log"
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
         
    utils.pipeline( architecture=architecture, 
                    config=config,
    )
    logger.info("\nProcessing completed successfully")

if __name__ == "__main__":

    config = utils.ModelConfig(
        description = "GfpBk8Caps",
        caps = ["CAP1", 
                "CAP2", 
                "CAP3", 
                "CAP4", 
                "CAP5", 
                "CAP6", 
                "CAP7", 
                "CAP8"],
        data_directory="data/pupil_training",
        task = ["rest",
                "dme",
                "monkey1", 
                "monkey2", 
                "monkey5", 
                "tp",
                "dmh"],
        additional_info=None,
        n_threads = 32,
        features_data_filename=None,
        
    )
        
    main(config)
