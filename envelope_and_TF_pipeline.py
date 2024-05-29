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
import eeg_research.preprocessing.tools.gradient_remover as GradientRemover
import eeg_research.preprocessing.tools.utils as utils
import numpy as np
import pickle

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
        elif key == 'run':
            value = int(value)
        elif key == 'desc':
            key = 'description'

        entities[key] = value
    return entities

class EEGfeatures:
    def __init__(self, raw: mne.io.Raw):
        self.raw = raw
        self.channel_names = raw.info['ch_names']
        self.times = raw.times
        self.frequencies = list()

    def _extract_envelope(self, frequencies: list[tuple[float,float]])-> np.ndarray:
        temp_envelopes_list = list()
        for band in frequencies:
            filtered = self.raw.copy().filter(*band)
            temp_envelopes_list.append(
                filtered.copy().apply_hilbert(envelope = True).get_data()
                )
        return np.stack(temp_envelopes_list, axis = -1)

    def extract_eeg_band_envelope(self: 'EEGfeatures') -> 'EEGfeatures':

        self.frequencies = [ 
                    (0.5, 4),
                    (4, 8),
                    (8, 13),
                    (13, 30),
                    (30, 40) 
                    ]
        self.feature = self._extract_envelope(self.frequencies)
        self.feature_info = "EEG bands envelopes"

        return self

    def extract_custom_band_envelope(self: 'EEGfeatures',
                                highest_frequency: int = 40,
                                lowest_frequency: int = 1, 
                                frequency_step: int = 1) -> 'EEGfeatures':
        for low_frequency in range(lowest_frequency, highest_frequency, frequency_step):
            high_frequency = low_frequency + frequency_step
            self.frequencies.append((low_frequency, high_frequency))

        self.feature = self._extract_envelope(self.frequencies)
        self.feature_info = f"""
        Custom bands envelopes 
        from {lowest_frequency} to {highest_frequency} Hz
        """

        return self

    def run_wavelets(self: 'EEGfeatures') -> 'EEGfeatures':

        self.frequencies = list(np.linspace(1,40,40))
        cycles = self.frequencies / 2
        self.feature = self.raw.copy().compute_tfr(freqs = self.frequencies, 
                                n_cycles = cycles,
                                method='morlet',
                                average = False,
                                return_itc = False,
                                n_jobs = -1)
        self.feature_info = """Morlet Time-Frequency Representation
        with 40 frequencies from 1 to 40 Hz number of cycles = frequency / 2"""
        
        return self
    
    def save(self, filename):
        param_to_save = {
            'channel_names': self.channel_names,
            'times': self.times,
            'frequencies': self.frequencies,
            'feature': self.feature,
            'feature_info': self.feature_info
        }
        print(f'saving into {filename}')
        with open(filename, 'wb') as file:
            pickle.dump(param_to_save, file)

def measure_gradient_time(raw, print_results = True):
    gradient_trigger_name = GradientRemover.extract_gradient_trigger_name(raw)
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    average_time_space = np.mean(np.diff(picked_events[:,0] * raw.info['sfreq']))
    std_time_space = np.std(np.diff(picked_events[:,0] * raw.info['sfreq']))
    if print_results:
        print(f'Average time space between gradient triggers: {average_time_space}')
        print(f'Standard deviation of time space between gradient triggers: {std_time_space}')
    return average_time_space

class TimeFrequency:
    def __init__(self, raw):
        self.raw = raw
        self.channel_names = raw.info['ch_names']
        self.times = raw.times

    
    def save(self, filename):
        param_to_save = {
            'channel_names': self.channel_names,
            'times': self.times,
            'power': self.power.get_data(),
            'frequencies': self.power.freqs,
        }
        print(f'saving into {filename}')
        with open(filename, 'wb') as file:
            pickle.dump(param_to_save, file)

def specific_crop(raw):
    gradient_time = measure_gradient_time(raw, print_results = False)
    gradient_trigger_name = GradientRemover.extract_gradient_trigger_name(raw)
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    picked_event_id = {gradient_trigger_name: event_id[gradient_trigger_name]}
    start = picked_events[0][0] * raw.info['sfreq']
    stop = picked_events[-1][0] * raw.info['sfreq'] + gradient_time
    cropped = raw.copy().crop(start, stop)
    return cropped

def Main():
    derivatives_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES/eeg_features_extraction')
    raw_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF')

    for filename in raw_path.iterdir():
        file_entities = parse_file_entities(filename)
        if file_entities['task'] == 'checker' or file_entities['task'] == 'rest':
            raw = mne.io.read_raw_edf(raw_path / filename, preload=True)
            bids_path = BIDSPath(**file_entities, 
                                root=derivatives_path,
                                datatype='eeg')
            bids_path.mkdir()
            
            bids_path.update(description = 'MorletTFR')
            tf_object = TimeFrequency(raw)

            saving_path = os.path.splitext(bids_path.fpath)[0] + '.pkl'
            tf_object.run_wavelets().save(saving_path)

            envelope = Envelope(raw)
            bids_path.update(description = 'EEGbandsEnvelopes')
            bands_envelope_filename = os.path.splitext(bids_path.fpath)[0] + '.pkl'
            envelope.extract_eeg_band_envelope().save(bands_envelope_filename)
            custom_envelope_filename = os.path.splitext(bids_path.fpath)[0] + '_custom.pkl'
            envelope.extract_custom_envelope().save(custom_envelope_filename)

if __name__ == '__main__':
    Main()