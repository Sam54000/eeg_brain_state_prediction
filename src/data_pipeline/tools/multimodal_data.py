from bids_explorer import architecture as arch
from src.data_pipeline.tools.configs import MultimodalConfig
from typing import Dict, List, Any
from pathlib import Path
from bids_explorer.paths.bids import BIDSPath
import os
import pickle
import numpy as np
from scipy.interpolate import CubicSpline

def print_shapes(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        title_parts = str(func.__name__).split("_")
        title = " ".join(title_parts).capitalize()
        print(f"\n{title}")
        for modality, data in result.items():
            print(f"    {modality}")
            print(f"            time shape: {data['time'].shape}")
            print(f"            data shape: {data['feature'].shape}")
            print(f"            mask shape: {data['mask'].shape}")
        return result
    return wrapper

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
    data: np.ndarray, 
    not_resampled_time: np.ndarray, 
    resampled_time: np.ndarray
) -> np.ndarray:

    interpolator = CubicSpline(not_resampled_time, data, axis=1)
    resampled = interpolator(resampled_time)
    return resampled

@print_shapes
def make_multimodal_dictionary(
    dict_modality: Dict[str, Path | str | None]
    ) -> Dict[str, Dict[str, Any]]:
    """ Make a dictionary containing dictionaries for several modalities. 
    
    Args:
        dict_modality (Dict[str, Path | str | None]): A dictionary containing
        the filenames for each modality.
    
    Returns:
        multimodal_dict (Dict[str, Dict[str, np.ndarray]]): A dictionary
            containing the data (values which are dictionaries) for each 
            modality (keys).
    """
    multimodal_dict = {}
    for modality, filename in dict_modality.items():
        if filename is None:
            continue
        try:
            with open(filename, "rb") as data_file:
                data = pickle.load(data_file)
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        multimodal_dict[modality] = data
        
    return multimodal_dict

@print_shapes
def resample_all(
    config: MultimodalConfig, 
    multimodal_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:

    resampled_multimodal = {}
    for modality, data in multimodal_dict.items():
        resampled_time = resample_time(
            data["time"],
            tr_value=config.tr_value,
            resampling_factor=config.resampling_factor,
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

        resampled_multimodal[modality] = {
            "time": resampled_time,
            "feature": resampled_features,
            "mask": resampled_mask,
            "labels": data["labels"],
            "feature_info": data["feature_info"],
            
        }
    return resampled_multimodal

@print_shapes
def trim_to_min_time(
    multimodal_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
    """ Trim each modality in order to have the same duration for all.
    
    Args:
        multimodal_dict (Dict[str, Dict[str, Any]]): Dictionary containing
            the data (values) for each modality (keys).
    
    Returns:
        trimed_multimodal (Dict[str, Dict[str, Any]]): Dictionary containing
            the trimmed data (values) for each modality (keys).
    """

    trimed_multimodal = {}
    min_time = min([data["time"][-1] for data in multimodal_dict.values()])
    min_length = np.argmin(
        abs(multimodal_dict["brainstates"]["time"] - np.floor(min_time))
    )

    for modality in multimodal_dict.keys():
        trimed_multimodal[modality] = {
                "time": multimodal_dict[modality]["time"][:min_length],
                "feature": multimodal_dict[modality]["feature"][:, :min_length, ...],
            "mask": multimodal_dict[modality]["mask"][:min_length],
        }

    return trimed_multimodal

def nice_print(
    subject: str,
    session: str,
    task: str,
    description: str,
    run: str,
    dict_modalities: Dict[str, Path | str | None]
    ) -> None:
    """ Print the filenames of the multimodal data in a nice format.
    
    Args:
        subject (str): The subject ID.
        session (str): The session ID.
        task (str): The task ID.
        description (str): The description of the task.
        run (str): The run ID.
        dict_modalities (Dict[str, Path | str | None]): A dictionary containing
            the filenames for each modality.
    """
    eeg_file = dict_modalities['eeg']
    brainstates_file = dict_modalities['brainstates']
    eyetracking_file = dict_modalities['eyetracking']
    print(f"└── Session: {session}")
    print(f"    └── Run: {run}")
    print(f"        ├── EEG file        : {eeg_file}")
    print(f"        ├── brainstates file: {brainstates_file }")
    print(f"        └── eyetracking file: {eyetracking_file }")

def collect_filenames(
    subject: str,
    session: str,
    task: str,
    run: str,
    config: MultimodalConfig,
    data_architecture: arch.BidsArchitecture,
    modalities: List[str],
) -> Dict[str, Path | str | None] | None:
    """Collect the filenames of the multimodal data.

    This function will generate a dictionary listing the filenames for
    each modality like a catalog for a specific subject, session, task and run.
    
    Args:
        subject (str): The subject ID.
        session (str): The session ID.
        task (str): The task ID.
        run (str): The run ID.
        config (MultimodalConfig): The configuration object.
        data_architecture (arch.BidsArchitecture): The architecture of the
        data.
    Returns:
        Dict[str, Path | str | None]: A dictionary containing the filenames of 
        the multimodal data (values) for each modality (keys).
    """

    selection = data_architecture.select(
        subject = subject, 
        task = task,
        session = session,
        run = run,
        datatype = modalities,
        suffix = modalities,
        extension = ".pkl"
    )
    if selection.database.empty:
        return None

    modalities = selection.suffixes

    if len(modalities) != len(modalities):
        return None
        
    combined_selection = combine_architectures(selection, modalities, config)

    dict_modality = {
        modality: combined_selection.database['filename'].values[0] if not(combined_selection.database.empty) else None
        for modality in modalities
    }

    return dict_modality

def save(path: Path,
         dict_modalities: Dict[str, Path | str | None],
         config: MultimodalConfig) -> None:

    saving_path = os.fspath(path.fullpath).replace(
        path.description,
        path.description + config.additional_description
        )
    print(f"\nSaving to: {saving_path}")
    try:
        with open(saving_path, "wb") as saving_file:
            pickle.dump(dict_modalities, saving_file)
    except Exception as e:
        print(f"Error saving to {saving_path}: {e}")

def combine_architectures(selection: arch.BidsArchitecture, 
                          modalities: List[str],
                          config: MultimodalConfig) -> arch.BidsArchitecture:
    """Combine the architectures for the different modalities.
    
    The function will combine individual arch.BidsArchitecture objects for
    each modality in order to have one arch.BidsArchitecture object for all
    desired modalities for a specific subject, session, task and run.
    
    Args:
        modalities (List[str]): The modalities to combine.

    Returns:
        BidsArchitecture: A dataframe containing the combined architectures.
    """

    combined_selection = []
    for modality in modalities:
        sub_selection = selection.select(
            datatype = modality,
            suffix = modality,
            description = config.brainstates_descriptions
        )
        if sub_selection.database.empty:
            return None
        combined_selection.append(sub_selection)

    combined_selection = sum(combined_selection)

    return combined_selection

