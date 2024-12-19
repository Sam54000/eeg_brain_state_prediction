#!/bin/bash
python src/extract_eeg_features.py > eeg_features_output_all.txt
python src/collect_multimodal_refact.py > multimodal_computing_all.txt