#!/bin/bash

# Run the feature selection script for each task and session
python /home/slouviot/01_projects/eeg_brain_state_prediction/src/eeg_brain_state_prediction/ml_pipeline/main_custom_envelope_features_selection_double_dipping_sub_level_sessions.py --task checker --session 02 --additional_info EegOnly
python /home/slouviot/01_projects/eeg_brain_state_prediction/src/eeg_brain_state_prediction/ml_pipeline/main_custom_envelope_features_selection_double_dipping_sub_level_sessions.py --task rest --session 02 --additional_info EegOnly
python /home/slouviot/01_projects/eeg_brain_state_prediction/src/eeg_brain_state_prediction/ml_pipeline/main_custom_envelope_features_selection_double_dipping_sub_level_sessions.py --task rest --session 02 --additional_info WithPupil


