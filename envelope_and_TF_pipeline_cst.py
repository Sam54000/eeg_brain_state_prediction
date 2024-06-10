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
import os
nthreads = "32" 
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
import mne
import bids
import mne_bids
from mne_bids import BIDSPath
from pathlib import Path
import matplotlib.pyplot as plt
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

def extract_eeg_only(raw: mne.io.Raw) -> mne.io.Raw:
    """Prepare the raw data for the processing.

    The function will set the channel types, the montage and the reference of the
    raw data.

    Args:
        raw (mne.io.Raw): The raw data that will be prepared.

    Returns:
        mne.io.Raw: The prepared raw data.
    """
    map = utils.map_channel_type(raw)
    raw.set_channel_types(map)
    montage = mne.channels.make_standard_montage('easycap-M1')
    raw.set_montage(montage)
    raw.pick_types(eeg = True)
    return raw

class BlinkRemover:
    def __init__(self, raw: mne.io.Raw, channels = ['Fp1', 'Fp2']):
        self.raw = raw
        self.channels = channels
    
    def _find_blinks(self):
        self.eog_evoked = mne.preprocessing.create_eog_epochs(self.raw, ch_name = self.channels).average()
        self.eog_evoked.apply_baseline((None, None))
        return self
    
    def plot_removal_results(self, saving_filename = None):
        figure = mne.viz.plot_projs_joint(self.eog_projs, self.eog_evoked)
        figure.suptitle("EOG projectors")
        if saving_filename:
            figure.savefig(saving_filename)
        plt.close()
    
    def plot_blinks_found(self, saving_filename = None):
        self._find_blinks()
        figure = self.eog_evoked.plot_joint(times = 0)
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
            self.raw, 
            n_eeg=1,
            reject=None,
            no_proj=True,
            ch_name = self.channels
        )
        self.blink_removed_raw = self.raw.copy()
        self.blink_removed_raw.add_proj(self.eog_projs).apply_proj()
        return self

class EEGfeatures:
    def __init__(self, raw: mne.io.Raw):
        self.raw = raw
        self.channel_names = raw.info['ch_names']
        self.frequencies = list()
    

    def _extract_envelope(self, frequencies: list[tuple[float,float]])-> np.ndarray:
        temp_envelopes_list = list()
        for band in frequencies:
            filtered = self.raw.copy().filter(*band)
            envelope = filtered.copy().apply_hilbert(envelope = True )
            temp_envelopes_list.append(envelope.get_data())
        self.times = envelope.times
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
        )
        
        self.times = self.raw.times
        self.feature = time_frequency_representation.get_data()
        self.feature_info = """Morlet Time-Frequency Representation
        with 40 frequencies from 1 to 40 Hz number of cycles = frequency / 2"""
        
        return self
    
    def save(self, filename):
        channel_info = extract_location(self.channel_names)
        param_to_save = {
            'channels_info': channel_info,
            'times': self.times,
            'frequencies': self.frequencies,
            'feature': self.feature,
            'feature_info': self.feature_info,
        }
        print(f'saving into {filename}')
        with open(filename, 'wb') as file:
            pickle.dump(param_to_save, file)

def loop(overwrite = True, 
         blank_run = True,
         remove_blinks = False,):
    raw_path = Path('/data2/Projects/NKI_RS2/MoBI/Extract_Mobi_Data/eeg_preprocessing_cst/data/annotated_eeg_data/calibration_data/eeg_data/')

    for filename in raw_path.rglob('*.fif'):
        try:
            individual_process(filename, 
                            overwrite = overwrite,
                            remove_blinks = remove_blinks,
                            blank_run=blank_run
                            )
        except Exception as e:
            print(f'___xxx___xxx___xxx___xxx___xxx___xxx___\n')
            print(f'Error with {filename}\n')
            print(e)
            print(f'\n___xxx___xxx___xxx___xxx___xxx___xxx___\n')

def individual_process(filename: Path, 
                    overwrite = True, 
                    remove_blinks = True,
                    blank_run = True):
    
    print('=====================================================\n')
    print(f'reading {filename}')

    derivatives_path = filename.parents[3] / 'eeg_features_extraction'
    file_entities = parse_file_entities(filename)
    bids_path = BIDSPath(**file_entities, 
                        root=derivatives_path,
                        datatype='eeg')
    

    if blank_run and remove_blinks:
        added_description = 'BlinksRemoved'
    elif blank_run and not remove_blinks:
        added_description = ''
    else:
        bids_path.mkdir()
        raw = mne.io.read_raw_fif(filename, preload=True)
        raw = extract_eeg_only(raw)

        if remove_blinks:
            added_description = 'BlinksRemoved'
            blink_remover = BlinkRemover(raw)
            blink_remover.remove_blinks()
            #fname_plot_blinks_found = Path(bids_path.update(description = 'BlinksFoundResults').fpath)
            #blink_remover.plot_blinks_found(
            #    saving_filename = fname_plot_blinks_found.with_suffix('.png')
            #                                )
            #fname_plot_blinks_results = Path(bids_path.update(description = 'BlinksRemovalResults').fpath)
            #blink_remover.plot_removal_results(
            #    saving_filename = fname_plot_blinks_results.with_suffix('.png')
            #)
            features_object = EEGfeatures(blink_remover.blink_removed_raw)
        else:
            added_description = ''
            features_object = EEGfeatures(raw)
            

    process_file_desc_pairs = {
        'run_wavelets': 'MorletTFR',
        'extract_eeg_band_envelope': 'EEGbandsEnvelopes',
        'extract_custom_band_envelope': 'CustomEnvelopes'
    }

    for process, file_description in process_file_desc_pairs.items():
        bids_path.update(description = file_description + added_description)
        saving_path = Path(os.path.splitext(bids_path.fpath)[0] + '.pkl')
        if not saving_path.exists() or overwrite:
            print(f'\tprocessing {process}')
            if blank_run:
                pass
            else: 
                features_object.__getattribute__(process)().save(saving_path)
            print(f'\tsaving into {saving_path}\n')
        else:
            continue


if __name__ == '__main__':
    for blink_removal in [True, False]:
        loop(overwrite = True, 
             blank_run = False, 
             remove_blinks = blink_removal)

# TODO The frequencies need a better handling because it changes datastructure
#      from list of tuple to numpy array. It could be better to keep it as np.array