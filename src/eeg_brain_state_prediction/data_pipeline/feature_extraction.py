import os
import numpy as np
import functools
from pathlib import Path

import pandas as pd
import mne
import bids_explorer.paths.bids as bids

from bids_explorer.utils.parsing import parse_bids_filename

import bids_explorer.architecture.architecture as arch
import eeg_brain_state_prediction.data_pipeline.tools.utils as utils
from eeg_brain_state_prediction.tools.configs import (
    PipelineConfig, 
    EegConfig,
    EegFeaturesConfig,
)

from eeg_brain_state_prediction.data_pipeline.tools.eeg import (
    EEGfeatures,
    crop,
    extract_envelope,
    extract_frequency_bands,
    extract_gfp,
    ssd_low_rank_factorization
)

from eeg_brain_state_prediction.data_pipeline.tools.utils import ProcessingError, log_execution

logger = utils.setup_logger(__name__, "feature_extraction_pipeline.log")

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
def pipeline(
    element: pd.Series,
    pipeline_config: PipelineConfig,
    eeg_config: EegConfig,
    eeg_features_config: EegFeaturesConfig,
    overwrite: bool,
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
    raw = mne.io.read_raw_edf(
        element["filename"],
        preload=True
        )
    
    eeg_features = EEGfeatures(
        raw=raw,
        feature_config=eeg_features_config,
        eeg_config=eeg_config,
    )

    tmin, tmax = utils.get_gradient_first_and_last_occurence(raw)
    
    eeg_features = crop(
        eeg_features=eeg_features,
        tmin=tmin,
        tmax=tmax,
        reset_time=True,
    )
    
    #eeg_features = extract_frequency_bands(eeg_features, eeg_features_config)
    #eeg_features = extract_envelope(eeg_features)
    #eeg_features = extract_gfp(eeg_features)
    #eeg_features.mask = np.ones_like(eeg_features.time, dtype=bool) #For some reason chang data doesn't give any good data but after visual review the data are very good so I pute everything to True.
    eeg_features = ssd_low_rank_factorization(eeg_features)
    eeg_features = extract_envelope(eeg_features)
    output_path = setup_path(
        filename = element["filename"],
        pipeline_config = pipeline_config,
        eeg_config = eeg_config
        )

    if output_path.fullpath.exists() and not(overwrite):
        return

    else:   
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
        datatype="eeg",
        suffix="eeg",
        extension=".edf",
        inplace=True
        )

    for file_id, element in architecture:
        pipeline(
            element=element,
            pipeline_config=pipeline_config,
            eeg_config=eeg_config,
            eeg_features_config=eeg_features_config,
            overwrite=pipeline_config.overwrite,
        )
    
if __name__ == "__main__":
    pipeline_config = PipelineConfig(
        raw_path=Path("/data2/Projects/eeg_fmri_natview/chang_data/raw"),
        derivatives_path=Path("/data2/Projects/eeg_fmri_natview/chang_data/derivatives"),
        overwrite=True,
        tasks=["MeRest"],
    )
    eeg_config = EegConfig(
        sampling_rate_hz=250,
        montage="easycap-M1",
        low_frequency_hz=0.5,
        high_frequency_hz=40,
        description="SSDbandsEnv",
        channels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz']
    )
    #low_freq = np.arange(1,39)
    #high_freq = np.arange(3,41)
    #freqs = [couple for couple in zip(low_freq, high_freq)]
    #freqs.insert(0, (0.5,2))
    freqs = [
        (0.5,4),
        (4,8),
        (8,13),
        (13,30),
        (30,40)
    ]
    
    eeg_features_config = EegFeaturesConfig(
        frequencies=freqs,
    )
    main(pipeline_config, eeg_config, eeg_features_config)

