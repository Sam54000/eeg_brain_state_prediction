from bids_explorer.utils.parsing import parse_bids_filename
from bids_explorer.paths.bids import BIDSPath
import mne
import os
from pathlib import Path
import numpy as np
import src.data_pipeline.tools.utils as utils

def individual_process(
    filename: str,
    overwrite=True,
    remove_blinks=True,
    blank_run=True,
    derivatives_path=None,
):
    print("=====================================================\nFILE ENTITIES:")
    file_entities = parse_bids_filename(filename)
    print("\n".join([f"\t{key}: {value}" for key, value in file_entities.items()]))
    print(f"\nREADING FILE:\n\t{filename}")

    print("\nPROCESSING:")
    bids_path = BIDSPath(
        **file_entities,
        root=derivatives_path,
        datatype="eeg",
        extension=".edf",
    )

    bids_path.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_edf(filename, preload=True)
    raw = utils.extract_eeg_only(raw)

    if remove_blinks:
        blink_remover = utils.BlinkRemover(raw)
        blink_remover.remove_blinks()
        features_object = utils.EEGfeatures(blink_remover.blink_removed_raw)
        description = ""
    else:
        features_object = utils.EEGfeatures(raw)
        description = "Bk"

    process_file_desc_pairs = {
        "extract_raw": f"Raw{description}",
        "extract_gfp": f"Gfp{description}",
        "extract_eeg_band_gfp": f"BandsGfp{description}",
        "extract_custom_band_gfp": f"CustomGfp{description}",
        "extract_eeg_band_envelope": f"BandsEnv{description}",
        "extract_custom_band_envelope": f"CustomEnv{description}",
    }

    for process, file_description in process_file_desc_pairs.items():
        bids_path.update(description=file_description)
        saving_path = Path(os.path.splitext(bids_path.fpath)[0] + ".pkl")
        if not saving_path.exists() or overwrite:
            print(f"    {process.upper()}")
            if blank_run:
                pass
            else:
                features_object.__getattribute__(process)()
                features_object.annotate_artifacts()
                if any(np.isnan(features_object.feature.flatten())):
                    raise Exception("ERROR: NAN GENERATED")
                features_object.save(saving_path)
        else:
            continue

def extract_eeg_features(
    overwrite=True,
    tasks =["rest", "checker"],
    blank_run=True,
    remove_blinks=False,
    derivatives_path=None,
):
    if blank_run:
        print("\n!!!!!! BLANK RUN !!!!!\n")
    raw_path = Path("/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BVA_GR_CB_BK_NOV2024")
    if remove_blinks:
        description = "GdCb"
    else:
        description = "GdCbBk"

    files = glob.glob(str(raw_path) + f"/sub-*_ses-*_task-*_run-*_desc-{description}_eeg.edf")

    for task in tasks:
        files = Path(raw_path).rglob(
            f"sub-*_ses-*_task-{task}_run-*_desc-{description}_eeg.edf"
        )
        
        for filename in files:

            individual_process(
                filename,
                overwrite=overwrite,
                remove_blinks=remove_blinks,
                blank_run=blank_run,
                derivatives_path=derivatives_path,
            )
