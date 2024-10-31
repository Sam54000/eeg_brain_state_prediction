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

Rule: EEG should always be a 3D array where the first dimension is the channels
the second dimension is the time and the third dimension the frequencies.
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
import eeg_research.preprocessing.pipelines.eeg_preprocessing_pipeline as pipe
import eeg_research.preprocessing.tools.artifacts_annotator as annotator
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
        try:
            key, value = entity.split('-')
        except:
            continue

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
            n_eeg=2,
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
        self.croping_values = specific_crop(raw,
                                            margin = -1,
                                            return_time = True)
    

    def extract_raw(self) -> 'EEGfeatures':
        self.time = self.raw.copy().crop(*self.croping_values).times
        self.feature = self.raw.copy().crop(*self.croping_values).get_data()
        self.feature = np.expand_dims(self.feature, axis = 2)
        self.frequencies = None
        self.feature_info = "Raw EEG signal"
        return self
    
    def extract_gfp(self) -> 'EEGfeatures':
        self.time = self.raw.copy().crop(*self.croping_values).times
        self.feature = np.expand_dims(
                np.var(
                    self.raw.copy().crop(*self.croping_values).get_data(),
                    axis = 0
                ),
                axis = 0
        )
        self.feature = np.expand_dims(self.feature, axis = 2)
        
        self.frequencies = None
        self.feature_info = "GFP of EEG signal"
        return self
            
    def _extract_envelope(self, frequencies: list[tuple[float,float]])-> 'EEGfeatures':
        temp_envelopes_list = list()
        for band in frequencies:
            filtered = self.raw.copy().filter(*band)
            envelope = filtered.copy().apply_hilbert(envelope = True ).crop(
                *self.croping_values
            )
            temp_envelopes_list.append(envelope.get_data())
        self.time = envelope.times
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
        for low_frequency in range(lowest_frequency, 
                                   highest_frequency, 
                                   frequency_step):
            high_frequency = low_frequency + frequency_step
            self.frequencies.append((low_frequency - 1, high_frequency))

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
        time_frequency_representation = self.raw.copy().crop(
            *self.croping_values
            ).compute_tfr(
            freqs = self.frequencies, 
            n_cycles = cycles,
            method='morlet',
            n_jobs = -1,
            reject_by_annotation=False
        )
    
        self.time = time_frequency_representation.times
        data_array = time_frequency_representation.get_data()
        self.feature = np.moveaxis(data_array, 2, 1)
        self.feature_info = """Morlet Time-Frequency Representation
        with 40 frequencies from 1 to 40 Hz number of cycles = frequency / 2"""
        
        return self
    
    def annotate_artifacts(self):
        annotator_instance = annotator.ZscoreAnnotator(
            self.raw.copy().crop(*self.croping_values)
            )
        annotator_instance.detect_muscles(filter_freq=(30, None)
                                          ).detect_other().merge_annotations()
        annotator_instance.compute_statistics().print_statistics()
        annotator_instance.generate_mask()
        self.mask = annotator_instance.mask
        
    def save(self, filename):
        channel_info = extract_location(self.channel_names)
        param_to_save = {
            'time': self.time,
            'labels':{'channels_info': channel_info,
                      'frequencies': self.frequencies,
            },
            'feature': self.feature,
            'feature_info': self.feature_info,
            'mask': self.mask
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

def specific_crop(raw: mne.io.Raw, 
                  margin: int = 1,
                  return_time = False) -> mne.io.Raw | tuple[float,float]:
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
    gradient_time = measure_gradient_time(raw, print_results = False)
    gradient_trigger_name = utils.extract_gradient_trigger_name(raw)
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    start = picked_events[0][0] / raw.info['sfreq'] - margin
    stop = (picked_events[-1][0] / raw.info['sfreq'] + gradient_time) + margin
    print(f'cropping from {start} to {stop}')
    if return_time:
        return (start,stop)
    else:
        cropped = raw.copy().crop(start, stop)
    return cropped

def individual_process(filename: str, 
                    overwrite = True, 
                    remove_blinks = True,
                    blank_run = True,
                    derivatives_path = None):
    
    print('=====================================================\n')
    print(f'reading {filename}')

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
        raw = mne.io.read_raw_edf(filename, preload=True)
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
        'extract_raw': 'raw',
        'extract_gfp': 'gfp',
        'extract_eeg_band_envelope': 'EEGbandsEnvelopes',
        #'extract_custom_band_envelope': 'CustomEnvelopes'
    }

    for process, file_description in process_file_desc_pairs.items():
        bids_path.update(description = file_description + added_description)
        saving_path = Path(os.path.splitext(bids_path.fpath)[0] + '.pkl')
        if not saving_path.exists() or overwrite:
            print(f'\tprocessing {process}')
            if blank_run:
                pass
            else: 
                features_object.__getattribute__(process)()
                features_object.annotate_artifacts()
                if any(np.isnan(features_object.feature.flatten())):
                    raise Exception('ERROR: NAN GENERATED')
                features_object.save(saving_path)
            print(f'\tsaving into {saving_path}\n')
        else:
            continue

def loop(overwrite = True, 
         task = ['rest', 'checker'],
         blank_run = True,
         remove_blinks = False,
         derivatives_path = None):
    raw_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF')

    for filename in raw_path.iterdir():
        #try:
        file_entities = parse_file_entities(filename)
        conditions = (file_entities['task'] in task and
                    file_entities['suffix'] == 'eeg' and
                    not 'GradientStep1' in filename.name)
        if conditions:
            individual_process(filename, 
                            overwrite = overwrite,
                            remove_blinks = remove_blinks,
                            blank_run=blank_run,
                            derivatives_path=derivatives_path
                            )
        #except Exception as e:
        #    print(f'___xxx___xxx___xxx___xxx___xxx___xxx___\n')
        #    print(f'Error with {filename}\n')
        #    print(e)
        #    print(f'\n___xxx___xxx___xxx___xxx___xxx___xxx___\n')


if __name__ == '__main__':
    blink_removal = True
    saving_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES/eeg_features_extraction/third_run')
    loop(overwrite = True, 
         task = ['rest','checker'],
         blank_run = False, 
         remove_blinks = blink_removal,
         derivatives_path = saving_path)

# TODO The frequencies need a better handling because it changes datastructure
#      from list of tuple to numpy array. It could be better to keep it as np.array