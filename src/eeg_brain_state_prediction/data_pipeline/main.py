from itertools import product
import os
from pathlib import Path
from eeg_brain_state_prediction.data_pipeline import feature_extraction, multimodal
import bids_explorer.architecture.architecture as arch
from eeg_brain_state_prediction.data_pipeline.tools.configs import (
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
            raw_path=pipeline_config.raw_path,
            subject=eeg_file["subject"],
            session = eeg_file["session"],
            task=eeg_file["task"],
            run=eeg_file["run"],
            pipeline_config=pipeline_config,
            eeg_config=multimodal_config.eeg,
            eeg_features_config=eeg_features_config,
        )
    
    multimodal_architecture = arch.BidsArchitecture(
        root = pipeline_config.derivatives_path
    )

    multimodal_architecture.select(
        subject = pipeline_config.subjects,
        task = pipeline_config.tasks,
        session = pipeline_config.sessions,
        run = pipeline_config.runs,
        inplace = True,
    )

    combination = product(
        multimodal_architecture.tasks, 
        multimodal_architecture.subjects,
        multimodal_architecture.sessions,
        multimodal_architecture.runs
    )

    for task, subject, session, run in combination:
        multimodal.pipeline(
            subject=subject,
            session = session,
            task = task,
            run = run,
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
        raw_path = Path("/data2/Projects/eeg_fmri_natview/raw"),
        derivatives_path = Path("/data2/Projects/eeg_fmri_natview/derivatives"),
        overwrite = False,
        code_root = Path(
            os.environ["HOME"],
            "01_projects",
            "eeg_brain_state_prediction",
        ),
        tasks= ["checker", "rest"],
        subjects= ["01"],
        sessions=["01"],
        runs=["01"],
    )

    eeg_config = EegConfig(
        description= "RawBk",
        sampling_rate_hz= 200,
        montage= "easycap-M1",
        low_frequency_hz= None,
        high_frequency_hz= None,
        channels= ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FPz', 'Fz', 'Cz', 'Pz', 'POz', 'Oz', 'FT9', 'FT10', 'TP9', 'TP10'],
    )

    eeg_features_config = EegFeaturesConfig(
        frequencies=[(None,None)],
    )

    brainstates_config = BrainstatesConfig(
        description = ["caps", 
                       "Cpca1054NrCombined",
                       "Cpca1054NeIndividual"],
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
                      "eeg",
                      "eyetracking"],
        brainstates = brainstates_config,
        eeg = eeg_config,
        eyetracking = eyetracking_config,
        additional_description = "NotResampled",
        )

    main(pipeline_config=pipeline_config,
         eeg_features_config=eeg_features_config,
         multimodal_config=multimodal_config)

