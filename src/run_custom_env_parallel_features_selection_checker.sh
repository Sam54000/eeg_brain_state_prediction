#!/bin/bash
subjects=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "20" "21" "22") 

# Path to Python script
script_path="/home/slouviot/01_projects/eeg_brain_state_prediction/src/train_custom_envelope_features_selection.py"

# Task name (update if needed)
task_name="checker"

description="CustomEnvBk"


# Export the required environment variables for Python
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the script in parallel for each subject
parallel -j 22 python3 $script_path --subject {} --task $task_name --desc $description ::: "${subjects[@]}"