import os
from scipy.interpolate import CubicSpline
import numpy as np
from eeg_research.system.bids_selector import BidsArchitecture,BidsDescriptor, BidsSelector, BidsPath
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def resample_time(
    time: np.ndarray,
    tr_value: float = 2.1,
    resampling_factor: float = 8,
    units: str = "seconds",
) -> np.ndarray:
    """Resample the time points of the data to a desired number of time points

    Args:
        time (np.ndarray): The time points of the data
        tr_value (float): The frequency of the TR if the argument
                          units = seconds or the period of the TR if
                          the argument units = hertz
        resampling_factor (float): The factor by which the data should be
                                   resampled.
        units (str): The units of the TR value. It can be either 'seconds' or
                     'Hertz'

    Returns:
        pd.DataFrame: The resampled time points

    Note:
        The resampling factor is how the data are downsampled or upsampled.
        If the resampling factor is greater than 1, the data are upsampled else,
        data are downsampled.

        Example 1:
        If the period of the TR is 2 seconds and the resampling factor is 2,
        then data will be upsampled to 1 second period (so 1 Hz).

        Example 2:
        If the period of the TR is 2 seconds and the resampling factor is 0.5,
        then data will be downsampled to 4 seconds period (so 0.25 Hz).

        Example 3:
        If the frequency of the TR is 2 Hz and the resampling factor is 2,
        then data will be upsampled to 4 Hz (so 0.25 seconds period).
    """

    if any([tr_value, resampling_factor]):
        if "second" in units.lower():
            power_one = 1
        elif "hertz" in units.lower():
            power_one = -1

        increment_in_seconds = (tr_value**power_one) * (resampling_factor**-1)

        time_resampled = np.arange(
            time[0], time[-1], increment_in_seconds
        )

        return time_resampled
    else:
        raise ValueError("You must provide the TR value and the resampling factor")


def resample_data(
    data: np.ndarray, not_resampled_time: np.ndarray, resampled_time: np.ndarray
):
    interpolator = CubicSpline(not_resampled_time, data, axis=1)
    resampled = interpolator(resampled_time)
    return resampled


def trim_to_min_time(multimodal_dict: dict):
    min_time = min([data["time"][-1] for data in multimodal_dict.values()])
    min_length = np.argmin(
        abs(multimodal_dict["brainstates"]["time"] - np.floor(min_time))
    )

    for modality in multimodal_dict.keys():
        multimodal_dict[modality].update(
            {
                "time": multimodal_dict[modality]["time"][:min_length],
                "feature": multimodal_dict[modality]["feature"][:, :min_length, ...],
                "mask": multimodal_dict[modality]["mask"][:min_length],
            }
        )

    return multimodal_dict


def nice_print(subject,
               session,
               task,
               description,
               run,
               dict_modalities):
    eeg_file = dict_modalities['eeg']
    brainstates_file = dict_modalities['brainstates']
    eyetracking_file = dict_modalities['eyetracking']
    print(f"\n==================================================================")
    print(f"Subject: {subject} | Task: {task} | Description: {description}")
    print(f"└── Session: {session}")
    print(f"    └── Run: {run}")
    print(f"        ├── EEG file        : {eeg_file}")
    print(f"        ├── brainstates file: {brainstates_file }")
    print(f"        └── eyetracking file: {eyetracking_file }")

            
if __name__ == "__main__":
    root = Path("/data2/Projects/eeg_fmri_natview/derivatives")
    tasks = ['peer',
             'monkey1',
             'checker',
             'rest',
             'tp',
             'inscapes',
             'dme',
             'dmh',
             'monkey5',
             'monkey2',
    ]
    overwrite = True
    for task in tasks:
        print(task)
        architecture = BidsArchitecture(root = root,
                                        task = task)
        
        resampling_factor = 8
        subjects = architecture.database['subject'].unique()
        eeg_descriptions = [
            #"GfpBk",
            #"CustomGfpBk",
            #"BandsGfpBk",
            #"BandsEnvBk",
            #"CustomEnvBk",
            "IscBandsEnvBk",
        ]
        
        multimodal = {}
        

        for subject in subjects:
            selection = architecture.copy().select(subject = subject, 
                                            datatype = ['brainstates','eeg','eyetracking'],
                                            suffix = ['brainstates','eeg','eyetracking'],
                                            extension = ".pkl")
            descriptor = selection.report()
            modalities = descriptor.suffixs

            #Put a print here to evaluate session(s) and run(s)

            for description in eeg_descriptions:
                print(modalities)
                if len(modalities) == 3:
                    for session in descriptor.sessions:
                        for run in descriptor.runs:
                            path = BidsPath(root = root,
                                    subject = subject,
                                    session = session,
                                    datatype = "multimodal",
                                    task = task,
                                    run = run,
                                    description = f"{description}{resampling_factor}",
                                    suffix = "multimodal",
                                    extension = ".pkl")
                            if path.fullpath.exists() and not(overwrite):
                                continue
                            mother_selection = selection.copy().select(
                                subject = subject,
                                run = run,
                                session = session,
                                task = task,
                                extension = ".pkl"
                            )
                                
                            
                            bs_selection = mother_selection.copy().select(
                                datatype = "brainstates",
                                suffix = "brainstates",
                                description = "caps"
                            )
                            eye_selection = mother_selection.copy().select(
                                datatype = "eyetracking",
                                suffix = "eyetracking",
                            )
                            eeg_selection = mother_selection.copy().select(
                                datatype = "eeg",
                                suffix = "eeg",
                                description = description,
                            )

                            dict_modality = {
                                'brainstates': bs_selection.database['filename'].values[0] if not(bs_selection.database.empty) else 'Not Existing',
                                'eyetracking': eye_selection.database['filename'].values[0] if not(eye_selection.database.empty) else 'Not Existing',
                                'eeg': eeg_selection.database['filename'].values[0] if not(eeg_selection.database.empty) else 'Not Existing',
                            }
                                
                            nice_print(subject,
                                       session,
                                       task,
                                       description,
                                       run,
                                       dict_modality)
                            
                            if any([bs_selection.database.empty, 
                                    eye_selection.database.empty, 
                                    eeg_selection.database.empty]):
                                continue

                            temp_df = pd.concat([bs_selection.database, 
                                                 eye_selection.database, 
                                                 eeg_selection.database])

                            for _,row in temp_df.iterrows():
                                with open(row['filename'], "rb") as data_file:
                                    data = pickle.load(data_file)

                                resampled_time = resample_time(
                                    data["time"],
                                    tr_value=2.1,
                                    resampling_factor=resampling_factor,
                                )

                                resampled_features = resample_data(
                                    data=data["feature"],
                                    not_resampled_time=data["time"],
                                    resampled_time=resampled_time,
                                )

                                resampled_mask = resample_data(
                                    data=data["mask"],
                                    not_resampled_time=data["time"],
                                    resampled_time=resampled_time,
                                )

                                data.update(
                                    {
                                        "time": resampled_time,
                                        "feature": resampled_features,
                                        "mask": resampled_mask,
                                    }
                                )

                                multimodal[row['datatype']] = data

                            trimed_multimodal = trim_to_min_time(multimodal_dict=multimodal)
                            for modality, data in multimodal.items():
                                print(f"\n{str(modality).upper()}")
                                print(f"    Before Trimming")
                                print(f"        time shape: {data['time'].shape}")
                                print(f"        data shape: {data['feature'].shape}")
                                print(f"        mask shape: {data['mask'].shape}")

                                print(f"    After Trimming")
                                print(f"        time shape: {trimed_multimodal[modality]['time'].shape}")
                                print(f"        data shape: {trimed_multimodal[modality]['feature'].shape}")
                                print(f"        mask shape: {trimed_multimodal[modality]['mask'].shape}")


                            path = BidsPath(root = root,
                                    subject = subject,
                                    session = session,
                                    datatype = "multimodal",
                                    task = task,
                                    run = run,
                                    description = f"{description}{resampling_factor}",
                                    suffix = "multimodal",
                                    extension = ".pkl")
                            print(f"\nSaving to: {path.fullpath}")
                            with open(path.fullpath, "wb") as saving_file:
                                pickle.dump(trimed_multimodal, saving_file)
