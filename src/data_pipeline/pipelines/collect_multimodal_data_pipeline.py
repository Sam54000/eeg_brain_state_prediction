import os
from pathlib import Path
import pickle
import pandas as pd

import bids_explorer.architecture as arch
import bids_explorer.paths.bids as bids
import eeg_research.preprocessing.pipelines.eeg_preprocessing_pipeline as pipe
import eeg_research.preprocessing.tools.artifacts_annotator as annotator
import src.data_pipeline.tools.multimodal_data as multimodal_data
from bids_explorer.paths.bids import BIDSPath
from itertools import product

def main(config: MultimodalConfig):

    architecture = arch.BidsArchitecture(root = config.data_root)
    combination = product(
        config.tasks, 
        [architecture.subjects[0]],
        architecture.sessions,
        architecture.runs
    )

    for task, subject, session, run in combination:
        
        dict_modality = multimodal_data.collect_filenames(
            subject = subject,
            session = session,
            task = task,
            run = run,
            config = config,
            data_architecture = architecture,
            modalities = config.modalities
        )

        multimodal_data.nice_print(
            subject = subject,
            session = session,
            task = task,
            description = config.eeg_descriptions,
            run = run,
            dict_modalities = dict_modality
            )
        
        path = BIDSPath(
            root = config.data_root,
            subject = subject,
            session = session,
            datatype = "multimodal",
            task = task,
            run = run,
            description = f"{config.eeg_descriptions}"\
                f"{config.resampling_factor}",
            suffix = "multimodal",
            extension = ".pkl"
            )

        if path.fullpath.exists() and not(config.overwrite):
            continue

        multimodal_dict = multimodal_data.make_multimodal_dictionary(
            dict_modalities = dict_modality
            )
        trimed_multimodal = multimodal_data.trim_to_min_time(
            multimodal_dict = multimodal_dict
            )

        multimodal_data.save(
            path = path,
            multimodal_dict = trimed_multimodal,
            config = config
            )


if __name__ == "__main__":
    config = MultimodalConfig(
        data_root = "/data2/Projects/eeg_fmri_natview/derivatives/",
        overwrite = False,
        eeg_descriptions = "RawBk",
        brainstates_descriptions = "caps",
        tasks = ["checker"],
        n_threads = 32,
    )
    
    logger = utils.setup_logger(
        log_file="main_collect_multimodal_data_not_resampled.log"
        )
    logger.info(f"Starting main_collect_multimodal_data_not_resampled")
    main(config)
    logger.info(f"Finished main_collect_multimodal_data_not_resampled")