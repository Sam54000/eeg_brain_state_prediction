#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-05-28
# email: samuel DOT louviot AT nki DOT rfmh DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================================
"""Signal Envelope and Time-Frequency extraction pipeline.

This pipeline is used in the project of brain state prediction from EEG data in
collaboration with John Hopkins University. It is used to extract the envelope 
of the EEG signal and the time-frequency representation of the signal.
"""
import mne
import bids
import mne_bids
from mne_bids import BIDSPath
import os
from pathlib import Path
import eeg_research.preprocessing.tools.utils as utils
import numpy as np
import pickle
import re

def extract_number_in_string(string: str) -> int:
    """Extract digit values in a string.

    Args:
        string (str): The string for which the digits will be extracted.

    Returns:
        int: The extracted digits
    """
    temp = re.findall(r'\d+', string)
    number = list(map(int, temp))
    return number[0]

def extract_channel_laterality(channel: str) -> str:
    """Extract the laterality of the channel.
    
    According to the international eeg standard, the laterality of channels
    are defined by the number. If the number is even, the channel is on the right
    side of the head, if the number is odd, the channel is on the left side of the
    head. If the channel has the letter 'z' instead of a number, 
    the channel is located on the midline.

    Args:
        channel (str): The name of the channel.

    Returns:
        str: The laterality of the corresponding channel.
    """
    if 'z' in channel.lower():
        return 'midline'
    else:
        number = extract_number_in_string(channel)
        if number % 2 == 0:
            return 'right'
        else:
            return 'left'

def extract_channel_anatomy(channel: str) -> str:
    """Extract the anatomical location of the channel from its name.

    Args:
        channel (str): The name of the channel.

    Returns:
        str: The anatomical location of the channel.
    """
    letter_anatomy_relationship = {
        'F': 'frontal',
        'C': 'central',
        'P': 'parietal',
        'O': 'occipital',
        'T': 'temporal',
        'Fp': 'frontopolar',
        'AF': 'anterior-frontal',
        'FC': 'fronto-central',
        'CP': 'centro-parietal',
        'PO': 'parieto-occipital',
        'FT': 'fronto-temporal',
        'TP': 'temporo-parietal',
    }
    pattern = re.findall(r'[a-zA-Z]+', channel)[0]
    pattern = pattern.replace('z', '')
    return letter_anatomy_relationship.get(pattern)

def extract_location(channels: list[str]) -> dict[str, list[str | int]]:
    """Extract the location of the channels from their names.
    
    The location (anatomical region and laterality) of the channels are extracted
    from their names.

    Args:
        channels (list[str]): The names of the channels.

    Returns:
        dict[str, list[str | int]]: A dictionary containing the index of the channel,
                              the name of the channel, the anatomical region and 
                              the laterality.
    """
    location = {
        'index': list(),
        'channel_name': list(),
        'anatomy': list(),
        'laterality': list()
    }
    for channel in channels:
        if 'ecg' in channel.lower() or 'eog' in channel.lower():
            continue
        info = (
            channels.index(channel),
            channel,
            extract_channel_anatomy(channel),
            extract_channel_laterality(channel)
        )
        
        for key, value in zip(location.keys(), info):
            location[key].append(value)

    return location

def parse_file_entities(filename: str | os.PathLike) -> dict:
    file_only = Path(filename).name
    basename, extension = os.path.splitext(file_only)
    entities = dict()
    entities['extension'] = extension
    entities['suffix'] = basename.split('_')[-1]
    for entity in basename.split('_')[:-1]:
        key, value = entity.split('-')

        if key == 'sub':
            key = 'subject'
        elif key == 'ses':
            key = 'session'
        elif key == 'acq':
            key = 'acquisition'
        elif key == 'desc':
            key = 'description'

        entities[key] = value
    return entities

class EEGfeatures:
    def __init__(self, raw: mne.io.Raw):
        self.raw = raw
        self.channel_names = raw.info['ch_names']
        self.frequencies = list()
        
    def _extract_envelope(self, frequencies: list[tuple[float,float]])-> np.ndarray:
        temp_envelopes_list = list()
        for band in frequencies:
            filtered = self.raw.copy().filter(*band, verbose = 'CRITICAL')
            envelope = filtered.copy().apply_hilbert(envelope = True, 
                                                     verbose = 'CRITICAL')
            envelope_cropped = specific_crop(envelope, margin = 0)
            temp_envelopes_list.append(envelope_cropped.get_data())
        self.times = envelope_cropped.times
        self.feature = np.stack(temp_envelopes_list, axis = -1)
        return self

    def extract_eeg_band_envelope(self: 'EEGfeatures') -> 'EEGfeatures':

        self.frequencies = np.array([ 
                    (0.5, 4),
                    (4, 8),
                    (8, 13),
                    (13, 30),
                    (30, 40) 
                    ]
        )
    
        self._extract_envelope(self.frequencies)
        self.feature_info = "EEG bands envelopes"

        return self

    def extract_custom_band_envelope(self: 'EEGfeatures',
                                highest_frequency: int = 40,
                                lowest_frequency: int = 1, 
                                frequency_step: int = 1) -> 'EEGfeatures':
        self.frequencies = list()
        for low_frequency in range(lowest_frequency, highest_frequency, frequency_step):
            high_frequency = low_frequency + frequency_step
            self.frequencies.append((low_frequency, high_frequency))

        self._extract_envelope(self.frequencies)
        self.frequencies = np.array(self.frequencies)
        self.feature_info = f"""
        Custom bands envelopes 
        from {lowest_frequency} to {highest_frequency} Hz
        """

        return self

    def run_wavelets(self: 'EEGfeatures') -> 'EEGfeatures':

        self.frequencies = np.linspace(1,40,40)
        cycles = self.frequencies / 2
        time_frequency_representation = self.raw.copy().compute_tfr(
            freqs = self.frequencies, 
            n_cycles = cycles,
            method='morlet',
            n_jobs = -1,
            verbose = 'CRITICAL')
        cropped_time_frequency_representation = specific_crop(
            time_frequency_representation, 
            margin = 0
            )
        self.times = cropped_time_frequency_representation.times
        self.feature = cropped_time_frequency_representation.get_data()
        self.feature_info = """Morlet Time-Frequency Representation
        with 40 frequencies from 1 to 40 Hz number of cycles = frequency / 2"""
        
        return self
    
    def save(self, filename):
        param_to_save = {
            'channels_info': extract_location(self.channel_names),
            'times': self.times,
            'frequencies': self.frequencies,
            'feature': self.feature,
            'feature_info': self.feature_info,
        }
        print(f'saving into {filename}')
        with open(filename, 'wb') as file:
            pickle.dump(param_to_save, file)

def measure_gradient_time(raw, print_results = True):
    gradient_trigger_name = utils.extract_gradient_trigger_name(raw)
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    average_time_space = np.mean(np.diff(picked_events[:,0] / raw.info['sfreq']))
    std_time_space = np.std(np.diff(picked_events[:,0] / raw.info['sfreq']))
    if print_results:
        print(f'Average time space between gradient triggers: {average_time_space}')
        print(f'Standard deviation of time space between gradient triggers: {std_time_space}')
    return np.round(average_time_space,1)

def specific_crop(raw: mne.io.Raw, margin: int = 1) -> mne.io.Raw:
    """Crop the raw data to get only when fMRI gradient is on.
    
    The function take the time of occurence of the first and the last gradient
    trigger to get raw data only between these two triggers. It also add a margin
    to anticipate edge effect that would be cropped after processing.
    Args:
        raw (mne.io.Raw): _description_
        padding (int, optional): Add a margin in second from the first and the
                                 to anticipate process 
                                 that would induce an edge effects. Defaults to 1.

    Returns:
        mne.io.Raw: _description_
    """
    gradient_time = measure_gradient_time(raw, print_results = False)
    gradient_trigger_name = utils.extract_gradient_trigger_name(raw)
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    start = picked_events[0][0] / raw.info['sfreq'] - margin
    stop = (picked_events[-1][0] / raw.info['sfreq'] + gradient_time) + margin
    print(f'cropping from {start} to {stop}')
    cropped = raw.copy().crop(start, stop)
    return cropped

def Main(overwrite = True):
    derivatives_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES/eeg_features_extraction')
    raw_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF')

    for filename in raw_path.iterdir():
        file_entities = parse_file_entities(filename)
        try: #I put a temporary error handling because some files don't have 
             #the eeg suffix and throw an error. This is just temporary for
             #the sake of productivity
            condition_respected  = (
                 (file_entities.get('tast') == 'checker' 
                  or file_entities.get('task') == 'rest')
                 and not file_entities.get('description') == 'GradientStep1'
             )

            if condition_respected:
                raw = mne.io.read_raw_edf(raw_path / filename, preload=True)
                bids_path = BIDSPath(**file_entities, 
                                    root=derivatives_path,
                                    datatype='eeg')
                bids_path.mkdir()
                
                features_object = EEGfeatures(raw)

                process_file_desc_pairs = {
                    'run_wavelets': 'MorletTFR',
                    'extract_eeg_band_envelope': 'EEGbandsEnvelopes',
                    'extract_custom_band_envelope': 'CustomEnvelopes'
                }

                for process, file_description in process_file_desc_pairs.items():
                    bids_path.update(description = file_description)
                    saving_path = Path(os.path.splitext(bids_path.fpath)[0] + '.pkl')
                    if not saving_path.exists() or overwrite:
                        features_object.__getattribute__(process)().save(saving_path)
                    else:
                        continue
                    
        except Exception as e:
            raise e
        


if __name__ == '__main__':
    Main()

# TODO The frequencies need a better handling because it changes datastructure
#      from list of tuple to numpy array. It could be better to keep it as np.array