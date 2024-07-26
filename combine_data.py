# %%
from sklearn.linear_model import Ridge
import numpy as np
import os
from typing import Any
import pickle

reading_dir = '/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models/data_prep/prediction_model_data_eeg_features_v2/group_data_Hz-3.8/'
task = 'rest'

def parse_filename(filename: str | os.PathLike) -> dict[str,str]:
    """parse filename that are somewhat like BIDS but not rigoursly like it.

    Args:
        filename (str | os.PathLike): The filename to be parsed

    Returns:
        dict[str,str]: The filename parts
    """
    splitted_filename = filename.split('_')
    filename_parts = {}
    for part in splitted_filename:
        splitted_part = part.split('-')
        if splitted_part[0] in ['sub','ses','run','task']:
            label, value = splitted_part
            filename_parts[label] = value
        
    return filename_parts
sessions = ['01', '02']
subjects = [ '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
tasks = [ 'checker_run-01', 'rest_run-01' ]
subjects_number = []
def combine_data(reading_dir: str | os.PathLike, 
                 task: str) -> dict[str,Any]:
    big_data = {}
    for subject in subjects:
        big_data[f'sub-{subject}'] = {}
        for session in sessions:
            for task in tasks:
                filename = f"sub-{subject}_ses-{session}_task-{task}_multimodal_data.pkl"
                full_filename = os.path.join(reading_dir,filename)
                if os.path.isfile(full_filename):
                    with open(full_filename, 'rb') as file: 
                        data = pickle.load(file)
                    big_data[f'sub-{subject}'][f'ses-{session}'] = data
    return big_data
# %%
big_d = combine_data(reading_dir,tasks[0])
# %%
