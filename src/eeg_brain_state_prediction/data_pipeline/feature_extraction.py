import os
import functools
from pathlib import Path

import pandas as pd
import mne
import bids_explorer.paths.bids as bids

from bids_explorer.utils.parsing import parse_bids_filename

import bids_explorer.architecture.architecture as arch
import eeg_brain_state_prediction.data_pipeline.tools.utils as utils
from eeg_brain_state_prediction.data_pipeline.tools.configs import PipelineConfig, EegConfig, EegFeaturesConfig
from eeg_brain_state_prediction.data_pipeline.tools.eeg import (
    EEGfeatures,
    crop,
)

from eeg_brain_state_prediction.data_pipeline.tools.utils import ProcessingError, log_execution

logger = utils.setup_logger(__name__, "feature_extraction_pipeline.log")

def prepare_pipeline(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> None:
        path = bids.BidsPath(
            root = kwargs["raw_path"],
            subject = kwargs["subject"],
            session = kwargs["session"],
            task = kwargs["task"],
            run = kwargs["run"],
            datatype = "eeg",
            suffix = "eeg",
            extension = ".edf",
            )
        
        logger.info("Starting processing of file: %s", path.fullpath)
            
        if not os.path.exists(path.fullpath):
            raise FileNotFoundError(f"Input file not found: {path.fullpath}")
            
        output_path = setup_path(
            filename = path.fullpath,
            pipeline_config = kwargs["pipeline_config"],
            eeg_config = kwargs["eeg_config"]
            )

        #try:
        return func(*args, **kwargs, output_path=output_path)
        #except Exception as e:
        #    raise ProcessingError(e)
            
    return wrapper

def setup_path(filename: str | Path,
               pipeline_config: PipelineConfig,
               eeg_config: EegConfig,
               ) -> bids.BidsPath:
    file_entities = parse_bids_filename(filename)
    file_entities.update(extension=".pkl")
    file_entities.update(description=eeg_config.description)

    bids_path = bids.BidsPath(
        **file_entities,
        root=pipeline_config.derivatives_path,
    )

    if not bids_path.fullpath.parent.exists():
        bids_path.fullpath.parent.mkdir(parents=True, exist_ok=True)
    
    return bids_path

@log_execution(logger)
@prepare_pipeline
def pipeline(
    raw_path: Path,
    subject: str,
    session: str,
    task: str,
    run: str,
    pipeline_config: PipelineConfig,
    eeg_config: EegConfig,
    eeg_features_config: EegFeaturesConfig,
    output_path: bids.BidsPath,
) -> None:
    """Process individual EEG file for feature extraction
    
    Args:
        architecture_row (pd.Series): Row from the architecture DataFrame
        pipeline_config (PipelineConfig): Pipeline configuration
        eeg_config (EegConfig): EEG configuration
        eeg_features_config (EegFeaturesConfig): EEG features configuration
        
    Raises:
        ProcessingError: If processing fails
        FileNotFoundError: If input file doesn't exist
    """
    raw_path = bids.BidsPath(
        root = raw_path,
        subject = subject,
        session = session,
        task = task,
        run = run,
        datatype = "eeg",
        suffix = "eeg",
        extension = ".edf",
        )

    raw = mne.io.read_raw_edf(
        raw_path.fullpath,
        preload=True
        )
    
    eeg_features = EEGfeatures(
        raw=raw,
        feature_config=eeg_features_config,
        eeg_config=eeg_config,
    )
    
    eeg_features = crop(
        eeg_features=eeg_features,
        eeg_config=eeg_config,
    )

    eeg_features.save(output_path.fullpath)

def main(pipeline_config: PipelineConfig,
         eeg_config: EegConfig,
         eeg_features_config: EegFeaturesConfig,
         ) -> None:
    
    architecture = arch.BidsArchitecture(
        root = pipeline_config.raw_path,
    )

    architecture.select(
        subject=pipeline_config.subjects,
        task=pipeline_config.tasks,
        inplace=True
        )

    for file_id, element in architecture:
        pipeline(
            raw_path=pipeline_config.raw_path,
            subject=element.subject,
            session=element.session,
            task=element.task,
            run=element.run,
            pipeline_config=pipeline_config,
            eeg_config=eeg_config,
            eeg_features_config=eeg_features_config,
        )
    
if __name__ == "__main__":
    pipeline_config = PipelineConfig(
        raw_path=Path("/data2/Projects/eeg_fmri_natview/raw"),
        derivatives_path=Path("/data2/Projects/eeg_fmri_natview/derivatives"),
        overwrite=True,
        tasks=["rest", "checker"],
    )
    eeg_config = EegConfig(
        sampling_rate_hz=200,
        montage="easycap-M1",
        low_frequency_hz=0.5,
        high_frequency_hz=40,
        channels=[
            'Fp1', 'Fp2','F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
            'F8', 'T7', 'T8', 'P7', 'P8', 'FPz', 'Fz', 'Cz', 'Pz', 'POz',
            'Oz', 'FT9', 'FT10', 'TP9', 'TP10',
        ],
    )
    eeg_features_config = EegFeaturesConfig(
        frequencies=[(0.5, 40)],
    )
    main(pipeline_config, eeg_config, eeg_features_config)

