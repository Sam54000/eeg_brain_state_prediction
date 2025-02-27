import functools
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
from eeg_brain_state_prediction.logger import setup_logger, log_execution

mne.set_log_level(verbose="ERROR", return_old_level=False, add_frames=None)

class DataPipelineError(Exception):
    """Base exception class for data pipeline errors"""
    pass

class ValidationError(DataPipelineError):
    """Raised when data validation fails"""
    pass

class ProcessingError(DataPipelineError):
    """Raised when data processing fails"""
    pass

class ConfigurationError(DataPipelineError):
    """Raised when configuration is invalid"""
    pass

def validate_data(data: np.ndarray, 
                 check_nan: bool = True, 
                 check_inf: bool = True,
                 check_shape: Optional[tuple] = None) -> None:
    """Validate numpy array data

    Args:
        data (np.ndarray): Input data to validate
        check_nan (bool): Whether to check for NaN values
        check_inf (bool): Whether to check for infinite values
        check_shape (Optional[tuple]): Expected shape of the data

    Raises:
        ValidationError: If validation fails
    """
    if check_shape and data.shape != check_shape:
        raise ValidationError(f"Data shape {data.shape} does not match expected shape {check_shape}")
    
    if check_nan and np.any(np.isnan(data)):
        raise ValidationError("Data contains NaN values")
    
    if check_inf and np.any(np.isinf(data)):
        raise ValidationError("Data contains infinite values")


class BlinkRemover:
    def __init__(self, raw: mne.io.Raw, channels=["Fp1", "Fp2"]):
        self.raw = raw
        self.channels = channels

    def _find_blinks(self):
        self.eog_evoked = mne.preprocessing.create_eog_epochs(
            self.raw, ch_name=self.channels
        ).average()
        self.eog_evoked.apply_baseline((None, None))
        return self



def extract_gradient_trigger_name(
    raw: mne.io.Raw, desired_trigger_name: str = "R128", on_missing: str = "raise"
) -> str | None:
    """Extract the name of the trigger for gradient artifact removal.

    Name of the gradient trigger can change across different paradigm,
    acquisition etc.

    Args:
        raw (mne.io.Raw): The raw object containing the EEG data.
        desired_trigger_name (str, optional): The theoretical name of the
                                            trigger or a substring.
                                            Defaults to "R128".
        on_missing (str, optional): What to do if the trigger is not found.
                                    Can be either "raise" or "warn" or "ignore".

    Returns:
        str | None: The name of the trigger for gradient artifact removal.

    Raises:
        Exception: No gradient trigger found.
    """
    annotations_names = np.unique(raw.annotations.description)
    for annotation_name in annotations_names:
        if desired_trigger_name.lower() in annotation_name.lower():
            return annotation_name

    if on_missing == "ignore":
        return None
    elif on_missing == "warn":
        warnings.warn("No gradient trigger found. Check the desired trigger name.")
        return None
    elif on_missing == "raise":
        raise Exception("No gradient trigger found. Check the desired trigger name.")
    else:
            return None
    
def measure_gradient_time(raw, print_results=True):
    gradient_trigger_name = extract_gradient_trigger_name(raw)
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    average_time_space = np.mean(np.diff(picked_events[:, 0] / raw.info["sfreq"]))
    std_time_space = np.std(np.diff(picked_events[:, 0] / raw.info["sfreq"]))
    if print_results:
        print(f"Average time space between gradient triggers: {average_time_space}")
        print(
            f"Standard deviation of time space between gradient triggers: {std_time_space}"
        )
    return np.round(average_time_space, 1)