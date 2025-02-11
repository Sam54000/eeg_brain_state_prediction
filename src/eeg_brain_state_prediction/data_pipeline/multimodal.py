import eeg_brain_state_prediction.data_pipeline.tools.configs as configs
import bids_explorer.architecture as arch
import eeg_brain_state_prediction.data_pipeline.tools.multimodal as multimodal
import bids_explorer.paths.bids as bids
from itertools import product
import eeg_brain_state_prediction.data_pipeline.tools.utils as utils
from pathlib import Path

def pipeline(
    subject: str,
    session: str,
    task: str,
    run: str,
    eeg_description: str,
    resampling_factor: float,
    overwrite: bool,
    derivatives_path: Path,
    modalities: list[str],
    multimodal_config: configs.MultimodalConfig,
    data_architecture: arch.BidsArchitecture,
    additional_description: str,
    ) -> None:

    dict_modality = multimodal.collect_filenames(
        subject = subject,
        session = session,
        task = task,
        run = run,
        multimodal_config = multimodal_config,
        data_architecture = data_architecture,
        modalities = modalities
        )

    multimodal.print_filenames(
        subject = subject,
        session = session,
        task = task,
        description = eeg_description,
        run = run,
        dict_modalities = dict_modality
        )

    if any((filename is None for filename in dict_modality.values())):
        return

    path = bids.BidsPath(
        root = derivatives_path,
        subject = subject,
        session = session,
        datatype = "multimodal",
        task = task,
        run = run,
        description = f"{eeg_description}"\
            f"{resampling_factor}",
        suffix = "multimodal",
        extension = ".pkl"
        )

    if path.fullpath.exists() and not(overwrite):
        return

    multimodal_dict = multimodal.make_multimodal_dictionary(
        dict_modality = dict_modality
        )
    trimed_multimodal = multimodal.trim_to_min_time(
        multimodal_dict = multimodal_dict
        )

    multimodal.save(
        path = path,
        multimodal_dict = trimed_multimodal,
        additional_description = additional_description
        )
    
def main(multimodal_config: configs.MultimodalConfig,
         pipeline_config: configs.PipelineConfig):

    architecture = arch.BidsArchitecture(
        root = pipeline_config.derivatives_path,
        )

    architecture.select(
        subject=pipeline_config.subjects,
        task=pipeline_config.tasks, 
        inplace=True
        )

    combination = product(
        architecture.tasks, 
        architecture.subjects,
        architecture.sessions,
        architecture.runs
    )

    for task, subject, session, run in combination:
        pipeline(
            subject = subject,
            session = session,
            task = task,
            run = run,
            eeg_description = multimodal_config.eeg.description,
            resampling_factor = multimodal_config.resampling_factor,
            overwrite = pipeline_config.overwrite,
            derivatives_path = pipeline_config.derivatives_path,
            modalities = multimodal_config.modalities,
            multimodal_config = multimodal_config,
            data_architecture = architecture,
            additional_description = multimodal_config.additional_description
            )
        
if __name__ == "__main__":

    pipeline_config = configs.PipelineConfig(
        overwrite = True,
        tasks = ["rest", "checker"],
        n_threads = 32,
    )

    brainstates_config = configs.BrainstatesConfig(
        description = ["caps", 
                       "Cpca1054NrCombined",
                       "Cpca1054NeIndividual"],
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
        additional_description = "NotResampled",
        )
    
    logger = utils.setup_logger(
        log_file="main_collect_multimodal_data_resampled.log"
        )
    logger.info("Starting main_collect_multimodal_data_resampled")
    main(multimodal_config, pipeline_config)
    logger.info("Finished main_collect_multimodal_data_resampled")