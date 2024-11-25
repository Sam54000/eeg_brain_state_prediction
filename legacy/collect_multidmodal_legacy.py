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

import legacy.s00_helper_functions_data_bva_edf3 as hf
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

base_out_dir = os.path.join(base_dir,'data_prep', 'prediction_model_data_eeg_features_checker')

out_dir = os.path.join(base_out_dir, f"group_data_Hz-{PREDHZ}")
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

#image_dir = os.path.join(out_dir, f"figures")
#if not os.path.exists(image_dir):
#   os.makedirs(image_dir)

#%% =======================================================================================

subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
sessions = [ '01', '02' ]
tasks = [
    #'tp_run-01',
    #'tp_run-02', 
    #'dme_run-01',
    #'dme_run-02',
    #'dmh_run-01',
    #'dmh_run-02',
    #'monkey1_run-01',
    #'monkey1_run-02',
    #'monkey2_run-01',
    #'monkey2_run-02', 
    #'monkey5_run-01', 
    #'monkey5_run-02',
    'checker_run-01',
    #'checker_run-02',
    'rest_run-01',
    #'rest_run-02', 
    #'inscapes_run-01'
    ]
bstate = 'cap_ts' # 'pca_cap_ts'

#%%
brainstate_dir = os.path.join(fmri_data_dir, bstate)


s = {
    'subjects': subjects,
    'sessions': sessions,
    'tasks': tasks
}
#%%
data_checker = pd.DataFrame()
i = 0
for sub in subjects:
    print(f"\n===============================================")
    for ses in sessions:
        for task in tasks:
            file_key   = f"| sub-{sub}_ses-{ses}_task-{task} |"
            separators = f"----------------------------"
            print(f"\n{separators.center(50)}")
            print(file_key.center(50,'-'))
            print(f"{separators.center(50)}")
            # ----------------------------------------------------------------------------
            # check if EEG or fMRI data exists
            
            try:
                _, existing_state_dict = hf.data_exists(sub, 
                                            ses, 
                                            task, 
                                            fmri_data_dir= fmri_data_dir,
                                            eyetrack_data_dir=eyetrack_data_dir, 
                                            eeg_proc_data_dir=eeg_proc_data_dir,
                                            #respiration_data_dir=respiration_data_dir,
                                            verbose = True
                                            )
                #Check the existence of data and log it
                data_keys_dict = {
                    'subject': sub,
                    'session': ses,
                    'task': task
                }
                data_keys_dict.update(existing_state_dict)
                data_checker = pd.concat([data_checker, 
                                          pd.DataFrame(data_keys_dict, index=[i])], 
                                         ignore_index=True)
                
                i += 1
                # ----------------------------------------------------------------------------
                # FMRI DATA

                brainstate_data = hf.get_brainstate_data_all(sub, 
                                                             ses, 
                                                             task, 
                                                             fmri_data_dir, 
                                                             TR_PERIOD)

                print(type(brainstate_data))
                if existing_state_dict['brainstates_data'] and \
                   existing_state_dict['pupil_data'] and \
                   isinstance(brainstate_data, pd.DataFrame) and\
                    existing_state_dict.get('eeg_data',False):
                       
                    resampled_time = hf.resample_time(brainstate_data['time'].values,
                                                    tr_value=TR_PERIOD,
                                                    resampling_factor=TR_PERIOD_DIVIDING)
                    brainstate_data_resampled = hf.resample_data(
                        brainstate_data,
                        time_resampled = resampled_time,
                        fill_nan = True
                        )
                    
                    bs_dict = hf.dataframe_to_dict(
                        brainstate_data_resampled,
                        column_names = [col 
                                        for col in brainstate_data_resampled.columns
                                        if hf.get_real_column_name(brainstate_data_resampled, 'time')
                                        not in col and 'tsMask' not in col],
                        info = 'brain states'
                    )
                    bs_dict.update({"mask":brainstate_data_resampled['tsMask'].values})
                    multimodal_data_dict = {'brainstates': bs_dict}

                    
                # ----------------------------------------------------------------------------
                # EYETRACKING DATA
                    eyetrack_data_file = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv")
                    pupil_data = pd.read_csv(eyetrack_data_file, sep='\t', index_col=0)
                    resampled_time_et = hf.resample_time(pupil_data['time'].values,
                                                        tr_value=TR_PERIOD,
                                                        resampling_factor=TR_PERIOD_DIVIDING)
                    pupil_data_resampled = hf.resample_data(pupil_data, 
                                                            time_resampled=resampled_time_et)
                    
                    pupil_data_resampled['first_derivative'] = np.diff(
                        pupil_data_resampled['pupil_size'].values, 
                        prepend=pupil_data_resampled['pupil_size'].values[0]
                        )
                    
                    pupil_data_resampled['second_derivative'] = np.diff(
                        pupil_data_resampled['pupil_size'].values, 
                        n=2, 
                        prepend=pupil_data_resampled['pupil_size'].values[:2]
                        )
                    
                    eyetrack_dict =  hf.dataframe_to_dict(
                        pupil_data_resampled,
                        column_names=[col for col in pupil_data_resampled.columns 
                                    if 'time' not in col.lower()
                                    and 'tmask' not in col.lower()],
                        info = 'Pupil data'
                    )
                    eyetrack_dict.update({'mask': pupil_data_resampled['tmask'].values})

                    
                    multimodal_data_dict['pupil'] = eyetrack_dict
                
                # ----------------------------------------------------------------------------
                # RESPIRATION DATA
                if existing_state_dict.get('respiration_data', False):
                    if (task[:2]=='tp'):
                        bstask = task
                    else:
                        bstask = task[:(len(task)-7)]
                    respiration_data_file = os.path.join(respiration_data_dir, f"sub-{sub}_ses-{ses}_task-{bstask}_resp_stdevs.csv")
                    respiration_data = pd.read_csv(respiration_data_file, sep=',', index_col=0)
                    resampled_time_resp = hf.resample_time(respiration_data['Time'].values,
                                                        tr_value=TR_PERIOD,
                                                        resampling_factor=TR_PERIOD_DIVIDING)
                    respiration_data_resampled = hf.resample_data(
                        respiration_data, 
                        time_resampled=resampled_time_resp)
                    respiration_data_dict =  hf.dataframe_to_dict(
                        respiration_data_resampled,
                        column_names=[col for col in respiration_data_resampled.columns
                                    if 'time' not in col.lower()],
                        info = 'Respiration data'
                    )
                    respiration_data_dict.update({
                        'mask': np.full_like(respiration_data_resampled['time'].values,
                                        True,
                                        dtype = bool)
                    })
                    multimodal_data_dict['respiration'] = respiration_data_dict
                    # change the column name StdDev_6s to respiration
                # ----------------------------------------------------------------------------
                # EEG DATA
                if  existing_state_dict['brainstates_data'] and \
                   existing_state_dict['pupil_data'] and \
                   isinstance(brainstate_data, pd.DataFrame) and\
                    existing_state_dict.get('eeg_data',False):
                    sub_dir_eeg = os.path.join(eeg_proc_data_dir, f"sub-{sub}", f"ses-{ses}", "eeg")
                    eeg_features = {
                        'EEGbandsEnvelopes': None,
                        'raw': None,
                        'gfp':None

                    }
                    if (task[:2]=='tp'):
                        bstask = task
                    else:
                        bstask = task[:(len(task)-7)]
                    
                    for key in eeg_features.keys():
                        filename = os.path.join(
                            sub_dir_eeg, 
                            f"sub-{sub}_ses-{ses}_task-{task}_desc-{key}BlinksRemoved_eeg.pkl")
                        
                        data = np.load(filename, allow_pickle=True)
                        resampled_eeg_time = hf.resample_time(
                            data['time'],
                            tr_value = TR_PERIOD,
                            resampling_factor = TR_PERIOD_DIVIDING
                            
                        )
                        eeg_features[key] = hf.resample_eeg_features(data, 
                                                                    resampled_eeg_time,
                                                                    verbose = True)

                    multimodal_data_dict|= eeg_features

                if all(existing_state_dict.values()):
                    end_times = []
                    print("\n")
                    for data_name, data in multimodal_data_dict.items():
                        data_shape = data["feature"].shape
                        time = data["time"]
                        text = f"Shape of {data_name}:"+ str(data_shape).rjust(41 - len(data_name))
                        print(text)
                        print(f"    End time: {time[-1]} seconds")
                        end_times.append(time[-1])

                    time_to_crop = np.min(end_times)
                    print(f"\nTime to crop: {time_to_crop}")
                    min_length = np.argmin(abs(
                        multimodal_data_dict['brainstates']['time'] - np.floor(time_to_crop)
                        ))
                    print(f"Index to crop: {min_length}\n")

                    for data_name, data in multimodal_data_dict.items():
                        key_list = ["time", "feature", "mask"]
                        print(data.keys())
                        for index, key_value in zip([0,1,0],key_list):
                            data_cropped = hf.crop_data(data[key_value],
                                                id_max = min_length,
                                                axis = index)
                            data.update({key_value:data_cropped})
                            text1 = f"Shape of {data_name} {key_value} before cropping:"
                            text = str(data_cropped.shape).rjust(41 - (len(data_name) + len(key_value)))
                            print(text1 + text)
                            print(f"    End time: {data['time'][-1]} seconds")

                        multimodal_data_dict.update({data_name:data})

                    key_name = f"sub-{sub}_ses-{ses}_task-{task}"
                    saving_name = os.path.join(out_dir, f"{key_name}_multimodal_data.pkl")
                    print(f"\nSaving file into {saving_name}")
                    with open(saving_name, "wb") as file:
                        pickle.dump( multimodal_data_dict, file)
                        
            except Exception as e:
                raise e
                #print(f"Error sub-{sub}_ses-{ses}_task-{task}: {e}")

#%%
data_checker.to_csv("data_checker.csv", index=False)