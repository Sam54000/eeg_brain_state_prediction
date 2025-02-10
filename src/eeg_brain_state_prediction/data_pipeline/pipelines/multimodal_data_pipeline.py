import eeg_brain_state_prediction.data_pipeline.tools.configs as configs
import bids_explorer.architecture as arch
import eeg_brain_state_prediction.data_pipeline.tools.multimodal_data as multimodal_data
import bids_explorer.paths.bids as bids
from itertools import product
import eeg_brain_state_prediction.data_pipeline.tools.utils as utils
import numpy as np
def main(multimodal_config: configs.MultimodalConfig,
         pipeline_config: configs.PipelineConfig):

    architecture = arch.BidsArchitecture(
        root = pipeline_config.derivatives_path,
        subject = "01",
        task = pipeline_config.tasks,
        )

    architecture.select(task=pipeline_config.tasks, inplace=True)
    combination = product(
        architecture.tasks, 
        architecture.subjects,
        architecture.sessions,
        architecture.runs
    )

    for task, subject, session, run in combination:
        
        dict_modality = multimodal_data.collect_filenames(
            subject = subject,
            session = session,
            task = task,
            run = run,
            multimodal_config = multimodal_config,
            data_architecture = architecture,
            modalities = multimodal_config.modalities
        )

        multimodal_data.nice_print(
            subject = subject,
            session = session,
            task = task,
            description = multimodal_config.eeg.description,
            run = run,
            dict_modalities = dict_modality
            )
        
        path = bids.BidsPath(
            root = pipeline_config.derivatives_path,
            subject = subject,
            session = session,
            datatype = "multimodal",
            task = task,
            run = run,
            description = f"{multimodal_config.eeg.description}"\
                f"{multimodal_config.resampling_factor}",
            suffix = "multimodal",
            extension = ".pkl"
            )

        if path.fullpath.exists() and not(pipeline_config.overwrite):
            continue

        multimodal_dict = multimodal_data.make_multimodal_dictionary(
            dict_modality = dict_modality
            )
        trimed_multimodal = multimodal_data.trim_to_min_time(
            multimodal_dict = multimodal_dict
            )

        multimodal_data.save(
            path = path,
            multimodal_dict = trimed_multimodal,
            multimodal_config = multimodal_config
            )


if __name__ == "__main__":

    pipeline_config = configs.PipelineConfig(
        overwrite = False,
        tasks = ["rest", "checker"],
        n_threads = 32,
    )

    brainstates_config = configs.BrainstatesConfig(
        description = ["caps"],
        brainstates = np.array([
            "CAP1", "CAP2", "CAP3", "CAP4", "CAP5", "CAP6", "CAP7", "CAP8"
        ])
    )

    eeg_config = configs.EegConfig(
        description = "RawBk",
        sampling_rate_hz = 200,
        montage = "easycap-M1",
        low_frequency_hz = 0.5,
        high_frequency_hz = 40,
    )

    eyetracking_config = configs.EyeConfig(
        description = None,
        features = ["pupil_dilation", "first_derivative", "second_derivative"],
    )

    multimodal_config = configs.MultimodalConfig(
        resampling_factor = 8,
        sampling_rate_hz = 3.8,
        tr_time_seconds = 2.1,
        modalities = ["brainstates", "eeg", "eyetracking"],
        brainstates = brainstates_config,
        eeg = eeg_config,
        eyetracking = eyetracking_config,
        )
    
    logger = utils.setup_logger(
        log_file="main_collect_multimodal_data_resampled.log"
        )
    logger.info(f"Starting main_collect_multimodal_data_resampled")
    main(multimodal_config, pipeline_config)
    logger.info(f"Finished main_collect_multimodal_data_resampled")