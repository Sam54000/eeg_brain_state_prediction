import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from bids_explorer import architecture as arch
from scipy.interpolate import CubicSpline

from eeg_brain_state_prediction.data_pipeline.tools.configs import MultimodalConfig
from eeg_brain_state_prediction.data_pipeline.tools.utils import (
    ProcessingError,
    ValidationError,
    log_execution,
    setup_logger,
    validate_data,
)

logger = setup_logger(__name__, "multimodal_data.log")

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

def validate_time_series(time: np.ndarray, data: np.ndarray) -> None:
    """Validate time series data
    
    Args:
        time (np.ndarray): Time points
        data (np.ndarray): Data values
        
    Raises:
        ValidationError: If validation fails
    """
    validate_data(time, check_nan=True, check_inf=True)
    validate_data(data, check_nan=True, check_inf=True)
    
    if time.ndim != 1:
        raise ValidationError("Time array must be 1-dimensional")
    if time.size != data.shape[1]:
        raise ValidationError(
            f"Time points ({time.size}) do not match data points ({data.shape[1]})"
        )

@log_execution(logger)
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
        np.ndarray: The resampled time points

    Raises:
        ValueError: If parameters are invalid
        ProcessingError: If resampling fails
    """
    try:
        if not any([tr_value, resampling_factor]):
            raise ValueError("You must provide the TR value and the resampling factor")
            
        if units.lower() not in ["seconds", "hertz"]:
            raise ValueError("Units must be either 'seconds' or 'Hertz'")

        validate_data(time, check_nan=True, check_inf=True)

        if "second" in units.lower():
            power_one = 1
        elif "hertz" in units.lower():
            power_one = -1

        increment_in_seconds = (tr_value**power_one) * (resampling_factor**-1)

        time_resampled = np.arange(
            time[0], time[-1], increment_in_seconds
        )

        return time_resampled
        
    except Exception as e:
        raise ProcessingError(f"Failed to resample time: {str(e)}")

@log_execution(logger)
def resample_data(
    data: np.ndarray, 
    not_resampled_time: np.ndarray, 
    resampled_time: np.ndarray
) -> np.ndarray:
    """Resample data using cubic spline interpolation
    
    Args:
        data (np.ndarray): Input data to resample
        not_resampled_time (np.ndarray): Original time points
        resampled_time (np.ndarray): Target time points
        
    Returns:
        np.ndarray: Resampled data
        
    Raises:
        ProcessingError: If resampling fails
    """
    try:
        validate_time_series(not_resampled_time, data)
        validate_data(resampled_time, check_nan=True, check_inf=True)

        interpolator = CubicSpline(not_resampled_time, data, axis=1)
        resampled = interpolator(resampled_time)
        
        validate_data(resampled, check_nan=True, check_inf=True)
        return resampled
        
    except Exception as e:
        raise ProcessingError(f"Failed to resample data: {str(e)}")

@print_shapes
@log_execution(logger)
def make_multimodal_dictionary(
    dict_modality: Dict[str, Union[Path, str, None]]
) -> Dict[str, Dict[str, Any]]:
    """Make a dictionary containing dictionaries for several modalities.
    
    Args:
        dict_modality: A dictionary containing the filenames for each modality.
    
    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the data for each modality.
        
    Raises:
        ProcessingError: If processing fails
    """
    multimodal_dict = {}
    
    for modality, filename in dict_modality.items():
        if filename is None:
            logger.debug("Skipping None filename for modality: %s", modality)
            continue
            
        try:
            logger.info("Loading data for modality: %s", modality)
            with open(filename, "rb") as data_file:
                data = pickle.load(data_file)
                
            if not isinstance(data, dict):
                raise ValidationError(f"Data for modality {modality} must be a dictionary")
                
            multimodal_dict[modality] = data
            logger.debug("Successfully loaded data for modality: %s", modality)
            
        except Exception as e:
            logger.error("Failed to load data for %s: %s", modality, str(e))
            raise ProcessingError(f"Failed to load data for modality {modality}: {str(e)}")
    
    return multimodal_dict

@print_shapes
def resample_all(
    multimodal_dict: Dict[str, Dict[str, Any]],
    multimodal_config: MultimodalConfig, 
    ) -> Dict[str, Dict[str, Any]]:

    resampled_multimodal = {}
    for modality, data in multimodal_dict.items():
        resampled_time = resample_time(
            data["time"],
            tr_value=multimodal_config.tr_time_seconds,
            resampling_factor=multimodal_config.resampling_factor,
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
    multimodal_config: MultimodalConfig,
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
        multimodal_config (MultimodalConfig): The configuration object.
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
        
    dict_modality = {}
    for modality in modalities:
        sub_selection = selection.select(
            datatype = modality,
            suffix = modality,
            description = getattr(multimodal_config, modality).description
        )
        temp = sub_selection.database.get('filename', None)
        if temp is None:
            dict_modality[modality] = None
        else:
            dict_modality[modality] = temp.values[0]

    return dict_modality

@log_execution(logger)
def save(
    path: Path,
    multimodal_dict: Dict[str, Dict[str, Any]],
    multimodal_config: MultimodalConfig
) -> None:
    """Save multimodal dictionary to file
    
    Args:
        path (Path): Path to save file
        multimodal_dict (Dict[str, Dict[str, Any]]): Data to save
        multimodal_config (MultimodalConfig): Configuration object
        
    Raises:
        ProcessingError: If saving fails
    """
    try:
        saving_path = os.fspath(path.fullpath).replace(
            path.description,
            path.description + multimodal_config.additional_description
        )
        
        logger.info("Saving data to: %s", saving_path)
        
        save_dir = Path(saving_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(saving_path, "wb") as saving_file:
            pickle.dump(multimodal_dict, saving_file)
            
        logger.info("Successfully saved data")
        
    except Exception as e:
        logger.error("Failed to save data: %s", str(e))
        raise ProcessingError(f"Failed to save data: {str(e)}")
