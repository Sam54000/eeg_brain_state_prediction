#!/bin/bash

script_path="/home/slouviot/01_projects/eeg_brain_state_prediction/src/train_custom_envelope.py"
task_name="checker"
descriptions=(
    "CustomEnvBkCpca1054ARcombined8",
    "CustomEnvBkCpca1054ARindividual8",
    "CustomEnvBkCpca1054BRcombined8",
    "CustomEnvBkCpca1054BRindividual8",
    "CustomEnvBkCpca1054NRcombined8",
    "CustomEnvBkCpca1054NRindividual8",
    "CustomEnvBkCpca1054Rawcombined8",
    "CustomEnvBkCpca1054Rawindividual8",
)
additional_info="WithPupil"

# Export the required environment variables for Python
export OMP_NUM_THREADS=35
export OPENBLAS_NUM_THREADS=35
export MKL_NUM_THREADS=35
export VECLIB_MAXIMUM_THREADS=35
export NUMEXPR_NUM_THREADS=35

# Run the script in parallel for each subject
 python3 $script_path --task $task_name --additional_info $additional_info --desc $descriptions {}::: "${descriptions[@]}"
python "/home/slouviot/01_projects/eeg_brain_state_prediction/labs/merge_dataframes.py" --task $task_name --desc $description --feat_selection 0
