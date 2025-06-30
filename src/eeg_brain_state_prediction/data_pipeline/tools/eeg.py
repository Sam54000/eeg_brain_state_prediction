import numpy as np
from mne.decoding import SSD
import mne
from scipy.signal import kaiserord, firwin
from dataclasses import dataclass
from typing import Optional, List
import eeg_brain_state_prediction.data_pipeline.tools.eeg_channels as eeg_channels
from typing import Optional
import pickle
from eeg_brain_state_prediction.tools.configs import (
    EegFeaturesConfig, 
    EegConfig,
)

import scipy.signal as signal
from eeg_brain_state_prediction.data_pipeline.tools.artifacts import Detector
from eeg_brain_state_prediction.data_pipeline.tools.utils import (
    log_execution,
    setup_logger,
)
from eeg_brain_state_prediction.data_pipeline.tools import features
logger = setup_logger(__name__, "feature_extraction.log")

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
        temp = eeg_features.raw.copy().filter(
            l_freq=low_frequency,
            h_freq=high_frequency
            )
        filtered_feature = temp.get_data()
        extracted_feature.append(filtered_feature)
    
    extracted_feature = np.stack(extracted_feature, axis=2)
    eeg_features.time = temp.times
    eeg_features.feature = extracted_feature
    eeg_features.feature_info.append(f"{len(feature_config.frequencies)} frequency bands extracted from {feature_config.frequencies[0][0]} to {feature_config.frequencies[-1][1]} Hz")
    return eeg_features

@log_execution(logger)
def crop(eeg_features: "EEGfeatures",
         tmin: Optional[float] = None,
         tmax: Optional[float] = None,
         reset_time: bool = True,
         ) -> "EEGfeatures":
    if tmin is not None:
        eeg_features.feature = eeg_features.feature[:, tmin:, :]
        eeg_features.mask = eeg_features.mask[tmin:]
        eeg_features.time = eeg_features.time[tmin:]
    if tmax is not None:
        eeg_features.feature = eeg_features.feature[:, :tmax, :]
        eeg_features.mask = eeg_features.mask[:tmax]
        eeg_features.time = eeg_features.time[:tmax]
    eeg_features.feature_info.append(
        f"Cropped from {eeg_features.time[0]}s to {eeg_features.time[-1]}s")
    if reset_time:
        eeg_features.time = eeg_features.time - eeg_features.time[0]
        eeg_features.feature_info.append(
            f"Reset time: {eeg_features.time[0]}s to {eeg_features.time[-1]}s")
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

def ssd_low_rank_factorization(eeg_features: "EEGfeatures") -> "EEGfeatures":
    extracted_features = []
    for frequencies in eeg_features.feature_config.frequencies:
        ssd = SSD(
            info=eeg_features.raw.info,
            reg="oas",
            sort_by_spectral_ratio=True,
            return_filtered=True,
            n_components=6,
            filt_params_signal=dict(
                l_freq=frequencies[0],
                h_freq=frequencies[1],
                l_trans_bandwidth=0.5,
                h_trans_bandwidth=0.5,
            ),
            filt_params_noise=dict(
                l_freq=frequencies[0]-1 if frequencies[0] > 1 else 0,
                h_freq=frequencies[1]+1,
                l_trans_bandwidth=0.5,
                h_trans_bandwidth=0.5,
            )
        )

        X = eeg_features.raw.copy().get_data()
        ssd.fit(X)
        extracted_features.append(ssd.apply(X))
    extracted_features = np.stack(extracted_features, 2)
    eeg_features.time = eeg_features.raw.times
    eeg_features.feature = extracted_features
    eeg_features.feature_info.append(
        f"{extracted_features.shape[2]} low rank factorisation through SSD "
        +f"from {eeg_features.feature_config.frequencies[0][0]} to "
        +f"{eeg_features.feature_config.frequencies[-1][1]} Hz"
        )
    
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
            self.feature_info = list()
            self.resample()
            map = eeg_channels.map_types(self.raw)
            self.raw.set_channel_types(map)
            montage = mne.channels.make_standard_montage(self.eeg_config.montage)
            self.raw.set_montage(montage)
            self.raw.pick_types(eeg=True)
            self.channel_selection = self._get_existing_channels()
            self.raw.pick(self.channel_selection)
            self.feature = np.expand_dims(
                self.raw.get_data(), 
                axis=2
            )
            self.frequencies = self.feature_config.frequencies
            #self._mask = self.annotate_artifacts(self.raw)
            self._mask = np.ones_like(self.time, dtype=bool) #For some reason chang data doesn't give any good data but after visual review the data are very good so I pute everything to True.
            self.labels = {
                "channels_info": eeg_channels.generate_dictionary(
                    self.channel_selection
                ),
                "frequencies": self.feature_config.frequencies,
            }

    @classmethod
    def from_raw(cls, raw: mne.io.Raw, feature_config: EegFeaturesConfig, eeg_config: EegConfig):
        return cls(raw = raw, 
                   feature_config = feature_config,
                   eeg_config = eeg_config)
    
    @property
    def mask(self):
        self._mask = np.ones_like(self.time, dtype=bool)
        return self._mask
    
    @mask.setter
    def mask(self, value):
        self._mask = value
        return self._mask
    
    def to_dict(self):
        return {
            "time": self.time,
            "labels": self.labels,
            "feature": self.feature,
            "feature_info": self.feature_info,
            "mask": self.mask,
        }

    def resample(self):
        self.feature_info.append(f"Resampled from {self.raw.info['sfreq']} Hz to {self.eeg_config.sampling_rate_hz} Hz")
        self.raw.resample(self.eeg_config.sampling_rate_hz)
        self.time = self.raw.times
        return self

    def _get_existing_channels(self):
        existing_channels = self.raw.info["ch_names"]
        if self.eeg_config.channels is not None:
            requested_channels = self.eeg_config.channels
            selection = [ch for ch in existing_channels if ch in requested_channels]
        else:
            selection = existing_channels
        return list(selection)
    
    def extract_gfp(self):
        gfp = np.std(self.feature, axis=0, keepdims=True)
        self.feature = gfp
        self.feature_info.append("GFP extracted")
        return self

    def annotate_artifacts(self, raw: mne.io.Raw):
        annotator_instance = Detector(raw)
        annotator_instance.detect_muscles(filter_freq=(100, None))
        annotator_instance.detect_other()
        annotator_instance.merge_annotations()
        annotator_instance.generate_mask()
        return annotator_instance.mask

    def save(self, filename):
        print(f"\nsaving into {filename}")
        print(self.to_dict())
        with open(filename, "wb") as file:
            pickle.dump(self.to_dict(), file)
