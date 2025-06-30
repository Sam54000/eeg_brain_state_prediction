from itertools import product
import os
from pathlib import Path
from eeg_brain_state_prediction.data_pipeline import feature_extraction, multimodal
import bids_explorer.architecture.architecture as arch
from eeg_brain_state_prediction.tools.configs import (
    PipelineConfig,
    EegConfig,
    EegFeaturesConfig,
    MultimodalConfig,
    BrainstatesConfig,
    EyeConfig
)

def main(
    pipeline_config: PipelineConfig,
    eeg_features_config: EegFeaturesConfig,
    multimodal_config: MultimodalConfig
) -> None:

    eeg_architecture = arch.BidsArchitecture(root = pipeline_config.raw_path)

    eeg_architecture.select(
        subject = pipeline_config.subjects,
        task = pipeline_config.tasks,
        session = pipeline_config.sessions,
        run = pipeline_config.runs,
        datatype = "eeg",
        suffix = "eeg",
        extension = ".edf",
        inplace = True,
    )

    for file_id, eeg_file in eeg_architecture:
        feature_extraction.pipeline(
            element=eeg_file,
            overwrite=pipeline_config.overwrite,
            pipeline_config=pipeline_config,
            eeg_config=multimodal_config.eeg,
            eeg_features_config=eeg_features_config,
        )
    
        multimodal_architecture = arch.BidsArchitecture(
            root = pipeline_config.derivatives_path
        )

        selection = multimodal_architecture.select(
            subject = eeg_file["subject"],
            task = eeg_file["task"],
            session = eeg_file["session"],
            acquisition = eeg_file["acquisition"],
        )

        multimodal.pipeline(
            element = eeg_file,
            eeg_description = multimodal_config.eeg.description,
            resampling_factor= multimodal_config.resampling_factor,
            overwrite=pipeline_config.overwrite,
            derivatives_path=pipeline_config.derivatives_path,
            modalities=multimodal_config.modalities,
            multimodal_config=multimodal_config,
            data_architecture=multimodal_architecture,
            additional_description=multimodal_config.additional_description
        )


if __name__ == "__main__":

    pipeline_config = PipelineConfig(
        n_threads = 32,
        raw_path = Path("/data2/Projects/eeg_fmri_natview/chang_data/raw"),
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

    eeg_config = EegConfig(
        description= "Gfp",
        sampling_rate_hz= 250,
        montage= "easycap-M1",
        low_frequency_hz= None,
        high_frequency_hz= None,
    )

    eeg_features_config = EegFeaturesConfig(
        frequencies=[(None,None)],
    )

    brainstates_config = BrainstatesConfig(
        description = ["caps"],
    )
        
    eyetracking_config = EyeConfig(
        description = None,
        features = ["pupil_dilation", 
                    "first_derivative",
                    "second_derivative"],
    )

    multimodal_config = MultimodalConfig(
        resampling_factor = 8,
        sampling_rate_hz = 3.8,
        tr_time_seconds = 2.1,
        modalities = ["brainstates", 
                      "eeg"],
        brainstates = brainstates_config,
        eeg = eeg_config,
        eyetracking = eyetracking_config, #When removing one modlity the default value is a String so it throws an error. To modify
        additional_description = "Caps",
        )

    main(pipeline_config=pipeline_config,
         eeg_features_config=eeg_features_config,
         multimodal_config=multimodal_config)

