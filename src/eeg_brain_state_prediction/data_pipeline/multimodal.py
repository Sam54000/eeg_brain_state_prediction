import os
from eeg_brain_state_prediction.tools.configs import (
    MultimodalConfig,
    PipelineConfig,
    BrainstatesConfig,
    EegConfig,
    EyeConfig,
)
import bids_explorer.architecture as arch
import eeg_brain_state_prediction.data_pipeline.tools.multimodal as multimodal
import bids_explorer.paths.bids as bids
from itertools import product
import eeg_brain_state_prediction.data_pipeline.tools.utils as utils
from pathlib import Path
import pandas as pd

def pipeline(
    element: pd.Series,
    eeg_description: str,
    resampling_factor: float,
    overwrite: bool,
    derivatives_path: Path,
    modalities: list[str],
    multimodal_config: MultimodalConfig,
    data_architecture: arch.BidsArchitecture,
    additional_description: str,
    ) -> None:
    """Run the multimodal pipeline

    Args:
        subject (str): Subject ID
        session (str): Session ID
        task (str): Task ID
        run (str): Run ID
        eeg_description (str): EEG description
        resampling_factor (float): Resampling factor
        overwrite (bool): Overwrite existing files
        derivatives_path (Path): Path to derivatives directory
        modalities (list[str]): List of modalities to process
        multimodal_config (MultimodalConfig): Multimodal configuration
        data_architecture (arch.BidsArchitecture): Bids architecture
        additional_description (str): Additional description
    """

    to_reject = ["atime",
                 "mtime",
                 "ctime",
                 "root",
                 "suffix",
                 "extension",
                 "datatype",
                 "filename",
                 "description"
                 ]
    kwargs = {e: val for e, val in element.items() 
              if (e not in to_reject and val is not None)}
    dict_modality = multimodal.collect_filenames(
        multimodal_config = multimodal_config,
        data_architecture = data_architecture,
        modalities = modalities,
        **kwargs
        )

    multimodal.print_filenames(
        dict_modalities = dict_modality,
        **kwargs
        )

    if any((filename is None for filename in dict_modality.values())):
        return

    path = bids.BidsPath(
        root = derivatives_path,
        datatype = "multimodal",
        description = f"{eeg_description}"\
            f"{resampling_factor}",
        suffix = "multimodal",
        extension = ".pkl",
        **kwargs
        )

    if path.fullpath.exists() and not(overwrite):
        return

    multimodal_dict = multimodal.make_multimodal_dictionary(
        dict_modality = dict_modality
        )

    resampled_multimodal = multimodal.resample_all(
        multimodal_dict = multimodal_dict,
        tr_time_seconds = multimodal_config.tr_time_seconds,
        resampling_factor = resampling_factor
        )

    trimed_multimodal = multimodal.trim_to_min_time(
        multimodal_dict = resampled_multimodal
        )

    multimodal.save(
        path = path,
        multimodal_dict = trimed_multimodal,
        additional_description = additional_description
        )
    
def main(multimodal_config: MultimodalConfig,
         pipeline_config: PipelineConfig):

    architecture = arch.BidsArchitecture(
        root = pipeline_config.derivatives_path,
        )

    architecture.select(
        subject=pipeline_config.subjects,
        task=pipeline_config.tasks, 
        inplace=True
        )
    for id, element in architecture:
        pipeline(
            element = element,
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

    pipeline_config = PipelineConfig(
        n_threads = 32,
        raw_path = Path("/data2/Projects/eeg_fmri_natview/chang_data/derivatives"),
        derivatives_path = Path("/data2/Projects/eeg_fmri_natview/chang_data/derivatives"),
        overwrite = False,
        code_root = Path(
            os.environ["HOME"],
            "01_projects",
            "eeg_brain_state_prediction",
        ),
        tasks= ["MeRest"],
        subjects= None,
        sessions= None,
        runs= None,
    )

    brainstates_config = BrainstatesConfig(
        description = ["caps"],
        
    )

    eeg_config = EegConfig(
        description = "SSDbandsEnv",
        sampling_rate_hz = 250,
        montage = "easycap-M1",
        low_frequency_hz = None,
        high_frequency_hz = None,
    )

    eyetracking_config = EyeConfig(
        description = None,
        features = ["pupil_dilation", "first_derivative", "second_derivative"],
    )

    multimodal_config = MultimodalConfig(
        resampling_factor = 8,
        sampling_rate_hz = 3.8,
        tr_time_seconds = 2.1,
        modalities = ["brainstates", "eeg"],
        brainstates = brainstates_config,
        eeg = eeg_config,
        eyetracking = eyetracking_config,
        additional_description = "Caps",
        )
    
    logger = utils.setup_logger(
        log_file="main_collect_multimodal_data_resampled.log"
        )
    logger.info("Starting main_collect_multimodal_data_resampled")
    main(multimodal_config, pipeline_config)
    logger.info("Finished main_collect_multimodal_data_resampled")