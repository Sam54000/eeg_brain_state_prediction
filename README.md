# Introduction
This dataset correspond to 3 different frequency features extracted from 
EEG data for checkerboard task and resting state. The features are:
- The dynamic of the different EEG bands: delta (0.5 to 4Hz), theta (4 to 8Hz), 
alpha (8 to 13Hz), beta (13 to 30Hz) and gamma (30 to 40Hz). 
- The dynamic of 40 frequency bands from 1Hz to 40Hz with a step of 1Hz.
- The time-frequency representation of the EEG from 1Hz to 40Hz.

## Dynamic of frequency bands
This feature was obtained by extracting the envelope of the narrow band filtered 
signal within. This process consisted of:
1. Filtering the EEG signal within desired frequency bands
2. Calculating the magnitude (absolute value) of the analytical signal obtained 
   from a Hilbert transform computed from the filtered signal. 
   
This feature is saved for each subject, task and session under this name pattern:
- For EEG frequency band envelopes
`sub-<subject#>_ses-<session#>_task<task_name>_desc-EEGbandsEnvelopes_eeg.pkl`
- For custom frequency band envelopes
`sub-<subject#>_ses-<session#>_task<task_name>_desc-CustomEnvelopes_eeg.pkl`

## Time-Frequency Representation (TFR)
The TFR has been obtained by Morlet wavelet transformation from 1Hz to 40Hz
(with a step of 1Hz) with a cycle of frequency / 2 for each frequency within the
desired range.

# Structure of the data
All files are saved as a pickle object. When deserializing (unpickling) you will
get a dictionnary with the following keys:
- channel_info
- times
- frequencies
- feature
- feature_info

## 