import functools
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np

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

def setup_logger(name: str = __name__, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Configure logging with timestamp and formatting

    Args:
        name (str): Logger name
        log_file (Optional[str]): Path to log file
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: Configured logger instance
    """
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_execution(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution with parameters and results

    Args:
        logger (Optional[logging.Logger]): Logger instance to use. If None, creates a new logger.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = setup_logger(func.__module__)

            logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

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

def set_thread_env(config) -> None:
    """Set environment variables for thread control with validation

    Args:
        config: Configuration object with n_threads attribute

    Raises:
        ConfigurationError: If thread configuration is invalid
    """
    if not hasattr(config, 'n_threads'):
        raise ConfigurationError("Configuration must have n_threads attribute")
    
    if not isinstance(config.n_threads, int) or config.n_threads < 1:
        raise ConfigurationError(f"Invalid n_threads value: {config.n_threads}")

    thread_vars = [
        "OMP_NUM_THREADS", 
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS"
    ]
    
    try:
        for var in thread_vars:
            os.environ[var] = str(config.n_threads)
    except Exception as e:
        raise ConfigurationError(f"Failed to set thread environment variables: {str(e)}")

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

    def plot_removal_results(self, saving_filename=None):
        figure = mne.viz.plot_projs_joint(self.eog_projs, self.eog_evoked)
        figure.suptitle("EOG projectors")
        if saving_filename:
            figure.savefig(saving_filename)
        plt.close()

    def plot_blinks_found(self, saving_filename=None):
        self._find_blinks()
        figure = self.eog_evoked.plot_joint(times=0)
        if saving_filename:
            figure.savefig(saving_filename)
        plt.close()

    def remove_blinks(self) -> mne.io.Raw:
        """Remove the EOG artifacts from the raw data.

        Args:
            raw (mne.io.Raw): The raw data from which the EOG artifacts will be removed.

        Returns:
            mne.io.Raw: The raw data without the EOG artifacts.
        """
        self.eog_projs, _ = mne.preprocessing.compute_proj_eog(
            self.raw, n_eeg=2, reject=None, no_proj=True, ch_name=self.channels
        )
        self.blink_removed_raw = self.raw.copy()
        self.blink_removed_raw.add_proj(self.eog_projs).apply_proj()
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