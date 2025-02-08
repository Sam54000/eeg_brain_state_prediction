
import os
nthreads = "32"
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
import glob
import pickle
import re
from itertools import product
from pathlib import Path

import bids_explorer.architecture as arch
import bids_explorer.paths.bids as bids
from bids_explorer.utils.parsing import parse_bids_filename
import eeg_research.preprocessing.tools.artifacts_annotator as annotator
import eeg_research.preprocessing.tools.utils as utils
import matplotlib.pyplot as plt
import mne
import eeg_channels
import mne_bids
import configs
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import logging
mne.set_log_level(verbose="ERROR", return_old_level=False, add_frames=None)

def setup_logger(log_file=None):
    """Configure logging with timestamp and formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_thread_env(config) -> None:
    """Set environment variables for thread control"""
    thread_vars = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
    ]
    for var in thread_vars:
        os.environ[var] = str(config.n_threads)

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


def measure_gradient_time(raw, print_results=True):
    gradient_trigger_name = utils.extract_gradient_trigger_name(raw)
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

def specific_crop(
    raw: mne.io.Raw, margin: int = 1, return_time=False
) -> mne.io.Raw | tuple[float, float]:
    """Crop the raw data to get only when fMRI gradient is on.

    The function take the time of occurence of the first and the last gradient
    trigger to get raw data only between these two triggers. It also add a margin
    to anticipate edge effect that would be cropped after processing.
    Args:
        raw (mne.io.Raw): _description_
        padding (int, optional): Add a margin in second from the first and the
                                 to anticipate process
                                 that would induce an edge effects. Defaults to 1.
        return_time (bool, optional): If True, the function will return the time
                                        of the first and the last gradient
                                        trigger instead of the raw object.

    Returns:
        mne.io.Raw: _description_
    """
    gradient_time = measure_gradient_time(raw, print_results=False)
    gradient_trigger_name = utils.extract_gradient_trigger_name(raw)
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    start = picked_events[0][0] / raw.info["sfreq"] - margin
    stop = (picked_events[-1][0] / raw.info["sfreq"] + gradient_time) + margin
    print(f"    cropping: from {start} to {stop} seconds\n")
    if return_time:
        return (start, stop)
    else:
        cropped = raw.copy().crop(start, stop)
    return cropped