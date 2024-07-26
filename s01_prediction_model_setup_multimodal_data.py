#%% =======================================================================================
#%load_ext autoreload
#%autoreload 2

import os
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.stats import zscore
import pickle

import matplotlib.pyplot as plt

import s00_helper_functions_data_bva_edf3 as hf
import importlib
importlib.reload(hf)

#%% =======================================================================================

base_dir = '/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models'
fmri_data_dir = '/data2/Projects/eeg_fmri_natview/data/fmriprep/derivatives'
#eeg_data_dir = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF'
eeg_proc_data_dir = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES/eeg_features_extraction/third_run'
eyetrack_data_dir = '/home/atambini/natview/eyetracking_resampled'
respiration_data_dir = '/home/thoppe/physio-analysis/resp-analysis/resp_stdevs'

#%% =======================================================================================

TR_PERIOD_DIVIDING = 8
TR_PERIOD = 2.1
PREDHZ = np.round(TR_PERIOD_DIVIDING/TR_PERIOD, 1)

#2.1/8 = 0.2625

base_out_dir = os.path.join(base_dir,'data_prep', 'prediction_model_data_eeg_features_v2')

out_dir = os.path.join(base_out_dir, f"group_data_Hz-{PREDHZ}")
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

dict_out_dir = os.path.join(base_out_dir, f"dictionary_group_data_Hz-{PREDHZ}")
if not os.path.exists(dict_out_dir):
   os.makedirs(dict_out_dir)

#image_dir = os.path.join(out_dir, f"figures")
#if not os.path.exists(image_dir):
#   os.makedirs(image_dir)

#%% =======================================================================================

subjects = [ '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
sessions = [ '01', '02' ] # 
#tasks = [ 'checker_run-01' ] #, 'tp_run-01', 'tp_run-02' ]
tasks = [ 'checker_run-01', 'rest_run-01' ]
bstate = 'cap_ts' # 'pca_cap_ts'

brainstate_dir = os.path.join(fmri_data_dir, bstate)


s = {
    'subjects': subjects,
    'sessions': sessions,
    'tasks': tasks
}

sub = '01' # 
ses = '01' # 01 021
task = 'checker_run-01' #'checker_run-01' 'rest_run-01' , 'tp_run-01', 'tp_run-02'

data_fmri = {}
data_eyetracking = {}
data_respiration = {}
data_eeg_bands_envelope = {}
data_eeg_custom_envelope = {}
data_eeg_morlet_tfr = {}

for sub in subjects:
    print(f"===============================================")
    for ses in sessions:
        for task in tasks:
            print(f"- {sub} - {ses} - {task}")
            # ----------------------------------------------------------------------------
            # check if EEG or fMRI data exists
            
            (has_fmri, 
             has_eeg_data, 
             has_pupil_data, 
             has_resp_data) = hf.check_data_exists(sub, 
                                                   ses, 
                                                   task, 
                                                   fmri_data_dir, 
                                                   eeg_proc_data_dir, 
                                                   eyetrack_data_dir, 
                                                   respiration_data_dir
                                                   )
             
            brainstate_data = hf.get_brainstate_data_all(sub, 
                                                         ses, 
                                                         task, 
                                                         fmri_data_dir, 
                                                         TR_PERIOD)
            
            #if not ( has_eeg & has_fmri & has_pd & has_resp ):

            #    print("---> not all features <---")
            #    continue
            
            # ----------------------------------------------------------------------------
            # FMRI DATA

            if all([isinstance(brainstate_data, pd.DataFrame), 
                    has_fmri,
                    has_pupil_data,
                    has_resp_data,
                    has_eeg_data]):

                resampled_time = hf.resample_time(brainstate_data['time'].values,
                                                  tr_value=TR_PERIOD,
                                                  resampling_factor=TR_PERIOD_DIVIDING)
                brainstate_data_resampled = hf.resample_data(
                    brainstate_data,
                    time_resampled = resampled_time,
                    fill_nan = True
                    )
                
                multimodal_data_dict = {'brainstates': hf.dataframe_to_dict(
                    brainstate_data_resampled,
                    column_names = [col 
                                    for col in brainstate_data_resampled.columns
                                    if hf.get_real_column_name(brainstate_data_resampled, 'time')
                                    not in col],
                    info = 'brain states'
                )
                }

                
            # ----------------------------------------------------------------------------
            # EYETRACKING DATA
                eyetrack_data_file = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv")
                pupil_data = pd.read_csv(eyetrack_data_file, sep='\t', index_col=0)
                pupil_data_resampled = hf.resample_data(pupil_data, time_resampled=resampled_time)
                multimodal_data_dict['pupil'] = hf.dataframe_to_dict(
                    pupil_data_resampled,
                    column_names=[col for col in pupil_data_resampled.columns 
                                  if 'time' not in col.lower()],
                    info = 'Pupil data'
                )
            
            # ----------------------------------------------------------------------------
            # RESPIRATION DATA
                if (task[:2]=='tp'):
                    bstask = task
                else:
                    bstask = task[:(len(task)-7)]
                respiration_data_file = os.path.join(respiration_data_dir, f"sub-{sub}_ses-{ses}_task-{bstask}_resp_stdevs.csv")
                respiration_data = pd.read_csv(respiration_data_file, sep=',', index_col=0)
                respiration_data_resampled = hf.resample_data(respiration_data, time_resampled=resampled_time)
                multimodal_data_dict['respiration'] = hf.dataframe_to_dict(
                    respiration_data_resampled,
                    column_names=[col for col in respiration_data_resampled.columns
                                  if 'time' not in col.lower()],
                    info = 'Respiration data'
                )
                # change the column name StdDev_6s to respiration

            # ----------------------------------------------------------------------------
            # EEG DATA

                sub_dir_eeg = os.path.join(eeg_proc_data_dir, f"sub-{sub}", f"ses-{ses}", "eeg")
                eeg_features = {
                    'EEGbandsEnvelopes': None,
                    'CustomEnvelopes': None,
                    'MorletTFR': None
                }
                if (task[:2]=='tp'):
                    bstask = task
                else:
                    bstask = task[:(len(task)-7)]
                
                for key in eeg_features.keys():
                    filename = os.path.join(
                        sub_dir_eeg, 
                        f"sub-{sub}_ses-{ses}_task-{task}_desc-{key}_eeg.pkl")
                    
                    print(key)
                    data = np.load(filename, allow_pickle=True)
                    eeg_features[key] = hf.resample_eeg_features(data, 
                                                                 resampled_time)

            # ----------------------------------------------------------------------------

                end_times = []
                multimodal_data_dict|= eeg_features

                for data_name, data in multimodal_data_dict.items():
                    data_shape = data["feature"].shape
                    time = data["time"]
                    print(f"Shape of {data_name} resampled: {data_shape}")
                    end_times.append(time[-1])

                time_to_crop = np.min(end_times)
                min_length = np.argmin(abs(
                    multimodal_data_dict['brainstates']['time'] - time_to_crop
                    ))

                for data_name, data in multimodal_data_dict.items():
                    for index, key_value in enumerate(["time", "feature"]):
                        data_cropped = hf.crop_data(data[key_value],
                                            id_max = min_length,
                                            axis = index)
                        data.update({key_value:data_cropped})
                        print(f"{data_name} {key_value} shape after cropping: {data_cropped.shape}")
                    
                    multimodal_data_dict.update({data_name:data})

                key_name = f"sub-{sub}_ses-{ses}_task-{task}"
                with open(os.path.join(dict_out_dir, f"{key_name}_multimodal_data.pkl"), "wb") as file:
                    pickle.dump( multimodal_data_dict, file)