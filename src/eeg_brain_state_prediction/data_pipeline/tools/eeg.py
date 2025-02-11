import numpy as np
import mne
from dataclasses import dataclass
from typing import Optional, List
import eeg_brain_state_prediction.data_pipeline.tools.eeg_channels as eeg_channels
from typing import Optional
import pickle
from eeg_brain_state_prediction.data_pipeline.tools.configs import EegFeaturesConfig, EegConfig
import scipy.signal as signal
from eeg_brain_state_prediction.data_pipeline.tools.artifacts import Detector
from eeg_brain_state_prediction.data_pipeline.tools.utils import (
    log_execution,
    setup_logger,
)
from eeg_brain_state_prediction.data_pipeline.tools import features
logger = setup_logger(__name__, "feature_extraction.log")

def apply_fir_filter(data: np.ndarray, 
                     sfreq: float, 
                     l_freq: Optional[float] = None, 
                     h_freq: Optional[float] = None, 
                     filter_length: str = 'auto', 
                     l_trans_bandwidth: float = 0.5, 
                     h_trans_bandwidth: float = 0.5) -> np.ndarray:
    """
    Apply a zero-phase FIR filter to a time series along the time dimension (axis 1).

    Args:
        data (np.ndarray): The time series data to filter with shape (..., n_times, ...).
        sfreq (float): Sampling frequency of the data.
        l_freq (float or None): Lower cutoff frequency. 
            If None, only a low-pass filter is applied (default is None).
        h_freq (float or None): Upper cutoff frequency. 
            If None, only a high-pass filter is applied (default is None).
        filter_length (str or int): Length of the FIR filter. 
            If 'auto', it will be determined automatically (default is 'auto').
        l_trans_bandwidth (float): Width of the transition band for the high-pass filter (in Hz).
        h_trans_bandwidth (float): Width of the transition band for the low-pass filter (in Hz).

    Returns:
        np.ndarray: The filtered time series with the same shape as input.

    Raises:
        ValueError: If cutoff frequencies exceed the Nyquist frequency (sfreq/2).
    """
    nyquist_freq = sfreq / 2

    if l_freq is not None and l_freq >= nyquist_freq:
        raise ValueError(f"Lower cutoff frequency ({l_freq} Hz) must be less than Nyquist frequency ({nyquist_freq} Hz)")
    if h_freq is not None and h_freq >= nyquist_freq:
        raise ValueError(f"Upper cutoff frequency ({h_freq} Hz) must be less than Nyquist frequency ({nyquist_freq} Hz)")

    if filter_length == 'auto':
        if l_freq is not None and h_freq is not None:
            filter_length = 'auto'
        elif l_freq is not None:
            filter_length = '10s'
        elif h_freq is not None:
            filter_length = '10s'
        else:
            raise ValueError("No filtering requested (both l_freq and h_freq are None).")

    if isinstance(filter_length, str):
        if filter_length.endswith('s'):
            filter_length = int(float(filter_length[:-1]) * sfreq)
        else:
            raise ValueError("filter_length must be 'auto' or a string ending with 's' (e.g., '10s').")

    if l_freq is not None and h_freq is not None:
        filt = signal.firwin(filter_length, [l_freq, h_freq], pass_zero=False, fs=sfreq,
                             window='hamming', scale=False)
    elif l_freq is not None:
        filt = signal.firwin(filter_length, l_freq, pass_zero=False, fs=sfreq,
                             window='hamming', scale=False)
    elif h_freq is not None:
        filt = signal.firwin(filter_length, h_freq, pass_zero=True, fs=sfreq,
                             window='hamming', scale=False)
    else:
        raise ValueError("No filtering requested (both l_freq and h_freq are None).")

    # Apply filter along time dimension (axis 1)
    filtered_data = np.apply_along_axis(
        lambda x: signal.filtfilt(filt, 1.0, x),
        axis=1,
        arr=data
    )

    return filtered_data

@log_execution(logger)
def extract_frequency_bands(
    eeg_features: "EEGfeatures",
    feature_config: EegFeaturesConfig
) -> "EEGfeatures":
    """Extract frequency bands from raw data
    
    Args:
        eeg_features (EEGfeatures): The eeg features
        frequencies (list[tuple[float, float]]): The frequencies to extract

    Returns:
        EEGfeatures: The eeg features with the extracted frequency bands
    """
    extracted_feature = []
    for low_frequency, high_frequency in feature_config.frequencies:
        filtered_feature = apply_fir_filter(eeg_features.feature,
                                            eeg_features.raw.info["sfreq"],
                                            l_freq=low_frequency, 
                                            h_freq=high_frequency
                                            )
        extracted_feature.append(filtered_feature)
    
    extracted_feature = np.stack(extracted_feature, axis=2)
    eeg_features.feature = extracted_feature
    eeg_features.feature_info.append(f"{len(feature_config.frequencies)} frequency bands extracted from {feature_config.frequencies[0][0]} to {feature_config.frequencies[-1][1]} Hz")
    return eeg_features

@log_execution(logger)
def crop(eeg_features: "EEGfeatures",
         eeg_config: EegConfig,
         ) -> "EEGfeatures":
    if eeg_config.tmin is not None:
        tmin_idx = np.argmin(np.abs(eeg_features.time - eeg_config.tmin))
        eeg_features.feature = eeg_features.feature[:, tmin_idx:]
        eeg_features.mask = eeg_features.mask[:, tmin_idx:]
        eeg_features.time = eeg_features.time[tmin_idx:]
    if eeg_config.tmax is not None:
        tmax_idx = np.argmin(np.abs(eeg_features.time - eeg_config.tmax))
        eeg_features.feature = eeg_features.feature[:, :tmax_idx]
        eeg_features.mask = eeg_features.mask[:, :tmax_idx]
        eeg_features.time = eeg_features.time[:tmax_idx]
    eeg_features.feature_info.append(
        f"Cropped from {eeg_features.time[0]}s to {eeg_features.time[-1]}s")
    return eeg_features

@log_execution(logger)
def extract_gfp(eeg_features: "EEGfeatures") -> "EEGfeatures":
    gfp = np.std(eeg_features.feature, axis=0, keepdims=True)
    eeg_features.feature = gfp
    eeg_features.feature_info.append("GFP extracted")
    return eeg_features

@log_execution(logger)
def extract_envelope(eeg_features: "EEGfeatures") -> "EEGfeatures":
    analytic_signal = signal.hilbert(eeg_features.feature, axis=1)
    envelope = np.abs(analytic_signal)
    eeg_features.feature = envelope
    eeg_features.feature_info.append("Envelope extracted")
    return eeg_features

@log_execution(logger)
def resample(eeg_features: "EEGfeatures",
             eeg_config: EegConfig,
             raw: mne.io.Raw,
             ) -> "EEGfeatures":
    """Resample the EEG data to a new sampling rate.
    
    This implementation follows MNE's approach using a FIR filter design for resampling.
    The function applies anti-aliasing filtering before resampling to prevent aliasing artifacts.
    
    Args:
        eeg_features (EEGfeatures): The EEG features object containing the data
        eeg_config (EegConfig): Configuration containing the target sampling rate
        
    Returns:
        EEGfeatures: The resampled EEG features
    """
    eeg_features.raw.resample(eeg_config.sampling_rate_hz)
    eeg_features.feature_info.append(f"Resampled from {eeg_features.sfreq} Hz to {eeg_config.sampling_rate_hz} Hz")
    return eeg_features

@dataclass
class EEGfeatures(features.BaseFeatures):
    raw: Optional[mne.io.Raw] = None
    feature_config: Optional[EegFeaturesConfig] = None
    eeg_config: Optional[EegConfig] = None
    time: Optional[np.ndarray] = None
    feature: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    feature_info: Optional[List[str]] = None
    labels: Optional[List] = None

    def __post_init__(self):
        super().__post_init__()
        conditions = (
            self.raw is not None,
            self.feature_config is not None,
            self.eeg_config is not None
        )
        if all(conditions):
            map = eeg_channels.map_types(self.raw)
            self.raw.set_channel_types(map)
            montage = mne.channels.make_standard_montage(self.eeg_config.montage)
            self.raw.set_montage(montage)
            self.raw.pick_types(eeg=True)
            channel_selection = self._get_existing_channels()
            self.feature = np.expand_dims(
                self.raw.get_data(picks=channel_selection), 
                axis=2
            )
            self.feature_info = list()
            self.channel_names = self.raw.info["ch_names"]
            self._resample()
            self.frequencies = self.feature_config.frequencies
            self.mask = self.annotate_artifacts(self.raw)
            self.time = self.raw.times
            self.labels = {
                "channels_info": eeg_channels.generate_dictionary(
                    self.channel_names
                ),
                "frequencies": self.feature_config.frequencies,
            }

    @classmethod
    def from_raw(cls, raw: mne.io.Raw, feature_config: EegFeaturesConfig, eeg_config: EegConfig):
        return cls(raw = raw, 
                   feature_config = feature_config,
                   eeg_config = eeg_config)
    
    
    def to_dict(self):
        return {
            "time": self.time,
            "labels": self.labels,
            "feature": self.feature,
            "feature_info": self.feature_info,
            "mask": self.mask,
        }

    def _resample(self):
        self.feature_info.append(f"Resampled from {self.raw.info['sfreq']} Hz to {self.eeg_config.sampling_rate_hz} Hz")
        self.raw.resample(self.eeg_config.sampling_rate_hz)
        self.time = self.raw.times
        return self

    def _get_existing_channels(self):
        existing_channels = set(self.raw.info["ch_names"])
        requested_channels = set(self.eeg_config.channels)
        selection = existing_channels.intersection(requested_channels)
        return list(selection)

    def annotate_artifacts(self, raw: mne.io.Raw):
        annotator_instance = Detector(raw)
        annotator_instance.detect_muscles(filter_freq=(30, None))
        annotator_instance.detect_other()
        annotator_instance.merge_annotations()
        annotator_instance.generate_mask()
        return annotator_instance.mask

    def save(self, filename):
        print(f"\nsaving into {filename}")
        with open(filename, "wb") as file:
            pickle.dump(self.to_dict(), file)
