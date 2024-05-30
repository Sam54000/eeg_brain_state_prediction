# Introduction

This dataset consists of three distinct types of frequency features extracted from EEG data, during both checkerboard tasks and resting states. The features include:

- Dynamics of EEG frequency bands: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), and Gamma (30-40Hz).
- Dynamics of 40 frequency bands ranging from 1Hz to 40Hz at 1Hz intervals.
- Time-frequency representation (TFR) of the EEG spanning 1Hz to 40Hz.

## Dynamics of Frequency Bands

This feature represents the envelope of the narrow band-filtered EEG signal, derived through the following process:

1. Filtering the EEG signal within the desired frequency bands.
2. Computing the magnitude (absolute value) of the analytical signal from a Hilbert transform of the filtered signal.

The resulting feature is structured and named according to the following pattern for each subject, task, and session:
- For EEG frequency band envelopes:
  `sub-<subject#>_ses-<session#>_task-<task_name>_desc-EEGbandsEnvelopes_eeg.pkl`
- For custom frequency band envelopes:
  `sub-<subject#>_ses-<session#>_task-<task_Name>_desc-CustomEnvelopes_eeg.pkl`

## Time-Frequency Representation (TFR)

The TFR is obtained using Morlet wavelet transformations across frequencies from 1Hz to 40Hz, with a frequency-dependent cycle of half the frequency value.

# Structure of the Data

All data are stored in pickle format. Deserializing the data yields a dictionary with the following keys:

## `channel_info`
Contains information about the anatomical positions of the EEG channels as a dictionary with keys:
- `index`: List of integers indicating the channel's position in the data.
- `channel_name`: List of strings naming each channel.
- `anatomy`: List of strings describing the anatomical locations (frontal, central, parietal, occipital, temporal, etc.).
- `laterality`: List of strings indicating the laterality of the electrodes (right, left, midline).

This structure allows for easy conversion into a pandas DataFrame, enabling the selection and grouping of channels by anatomical area and/or laterality.

## `times`
A one-dimensional numpy array representing the time in seconds for each data sample.

## `frequencies`
A one-dimensional numpy array denoting the frequencies of the extracted features.

## `feature`
A three-dimensional numpy array providing the values of the features across channels (1st Dimension), time (2nd Dimension), and frequency (3rd Dimension).

## `feature_info`
A string describing the feature corresponding to the file being read.