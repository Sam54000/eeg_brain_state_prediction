#%% =======================================================================================
#%load_ext autoreload
#%autoreload 2

import os
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.stats import zscore

import matplotlib.pyplot as plt

import s00_helper_functions_data_bva_edf5 as hf
import importlib
importlib.reload(hf)

#%% =======================================================================================

base_dir = '/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models_with_legacy_code'

fmri_data_dir = '/data2/Projects/eeg_fmri_natview/data/fmriprep/derivatives'
eeg_data_dir = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF'
eyetrack_data_dir = '/home/atambini/natview/eyetracking_resampled'
respiration_data_dir = '/home/thoppe/physio-analysis/resp-analysis/resp_stdevs'

#%% =======================================================================================

predHz = 3.8 #0.5 # 0.5

base_out_dir = os.path.join(base_dir,'data_prep', 'prediction_model_data_eyetracking_all_tasks_fixed')

out_dir = os.path.join(base_out_dir, f"group_data_Hz-{predHz}")
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

#image_dir = os.path.join(out_dir, f"figures")
#if not os.path.exists(image_dir):
#   os.makedirs(image_dir)

#%% =======================================================================================

subjects = [ '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
sessions = [ '01', '02' ] # 
tasks = [ 'checker_run-01', 'rest_run-01', 'inscapes_run-01', 
            'dme_run-01', 'dme_run-02', 'monkey1_run-01', 'monkey1_run-02', 'tp_run-01', 'tp_run-02', 
            'dmh_run-01', 'dmh_run-02', 'monkey2_run-01', 'monkey2_run-02', 'monkey5_run-01', 'monkey5_run-02' ]
#tasks = [ 'checker_run-01', 'rest_run-01' ]
bstate = 'cap_ts' # 'pca_cap_ts'

brainstate_dir = os.path.join(fmri_data_dir, bstate)

s = {
    'subjects': subjects,
    'sessions': sessions,
    'tasks': tasks
}

bold_tr = 2.1
pd_tr = 1.05

sub = '01' # 
ses = '01' # 01 021
task = 'inscapes_run-01' #'checker_run-01' 'rest_run-01' , 'tp_run-01', 'tp_run-02'

data_fmri = {}
data_eyetracking = {}
data_respiration = {}
for sub in subjects:
    print(f"===============================================")
    for ses in sessions:
        for task in tasks:
            print(f"- {sub} - {ses} - {task}")
            # ----------------------------------------------------------------------------
            # check if EEG or fMRI data exists
            
            has_fmri, has_eeg, has_pd, has_resp = hf.check_data_exists(sub, ses, task, fmri_data_dir, eeg_data_dir, eyetrack_data_dir, respiration_data_dir)

            if not ( has_fmri & has_pd ):
                print("---> not all three features <---")
                continue
            
            # ----------------------------------------------------------------------------
            # FMRI DATA

            bs_exists, bstate_data = hf.get_brainstate_data_all(sub, ses, task, fmri_data_dir, bold_tr)
            if not ( bs_exists ):
                print("-- no brainstates")
                continue

            FMRI_TARGET_RESAMPLED = []
            if bs_exists:
                FMRI_TARGET = bstate_data
                n_samples = np.round(bstate_data.shape[0] / ((1/bold_tr)/predHz)).astype('int') # tr 2.1sec to 1Hz - 1sec
                
                fmri_time = bstate_data['time'].to_numpy()
                if (predHz == 3.8):
                    inc = 0.2625
                    fmri_time_resampled = np.arange(fmri_time[0], fmri_time[-1]+inc, inc)
                elif (predHz == 0.5):
                    inc = bold_tr
                    fmri_time_resampled = np.arange(fmri_time[0], fmri_time[-1]+inc, inc)
                else:
                    fmri_time_resampled = np.linspace(fmri_time[0], fmri_time[-1], num=n_samples)
                
                bstate_data = bstate_data.fillna(0)
                FMRI_TARGET_RESAMPLED = pd.DataFrame()
                for col in bstate_data.columns:
                    tmp_spline = CubicSpline(fmri_time, bstate_data[col])
                    FMRI_TARGET_RESAMPLED[col] = tmp_spline(fmri_time_resampled)

                #data_fmri[f"sub-{sub}_ses-{ses}_task-{task}"] = FMRI_TARGET_RESAMPLED

            # ----------------------------------------------------------------------------
            # EYETRACKING DATA

            PUPIL_DATA_RESAMPLED = []
            if has_pd:
                eyetrack_data_file = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv")
                pd_df = pd.read_csv(eyetrack_data_file, sep='\t', index_col=0)

                n_samples = np.round(pd_df.shape[0] / ((1/pd_tr)/predHz)).astype('int')
                pd_time = pd_df['time'].to_numpy()
                pd_spline = CubicSpline(pd_time, pd_df['pupil_size'])
                #pd_time_resampled = np.linspace(pd_time[0], pd_time[-1], num=n_samples)
                pd_time_resampled = fmri_time_resampled
                pupil_size_resampled = pd_spline(pd_time_resampled)

                pd_X_spline = CubicSpline(pd_time, pd_df['X_position'])
                pd_X_position_resampled = pd_X_spline(pd_time_resampled)
                pd_Y_spline = CubicSpline(pd_time, pd_df['Y_position'])
                pd_Y_position_resampled = pd_Y_spline(pd_time_resampled)

                #pd_tmask = pd_df['tmask'].to_numpy().astype(float)
                pd_mask_spline = CubicSpline(pd_time, pd_df['tmask'])
                pd_tmask_resampled = pd_mask_spline(pd_time_resampled)
                
                PUPIL_DATA_RESAMPLED = pd.DataFrame(
                    data = np.vstack((pd_time_resampled, pupil_size_resampled, pd_X_position_resampled, pd_Y_position_resampled, pd_tmask_resampled)).T,
                    columns = ['time', 'pupil_size', 'X_position', 'Y_position', 'tmask'] )
                #data_eyetracking[f"sub-{sub}_ses-{ses}_task-{task}"] = PUPIL_DATA_RESAMPLED        

            # ----------------------------------------------------------------------------

            print(f"FMRI_TARGET_RESAMPLED {FMRI_TARGET_RESAMPLED.shape}")
            print(f"PUPIL_DATA_RESAMPLED {PUPIL_DATA_RESAMPLED.shape}")
            print(f"cropping it ...")
            #min_length = np.min( [ len(FMRI_TARGET_RESAMPLED), len(PUPIL_DATA_RESAMPLED), len(RESPIRATION_DATA_RESAMPLED) ] )
            max_time = np.min( [ fmri_time[-1], pd_time[-1]] )
            min_length = np.where((FMRI_TARGET_RESAMPLED['time'] < max_time) == 0)[0][0]

            FMRI_TARGET_RESAMPLED = FMRI_TARGET_RESAMPLED.loc[:min_length, :]
            PUPIL_DATA_RESAMPLED = PUPIL_DATA_RESAMPLED.loc[:min_length, :]

            data_fmri[f"sub-{sub}_ses-{ses}_task-{task}"] = FMRI_TARGET_RESAMPLED
            data_eyetracking[f"sub-{sub}_ses-{ses}_task-{task}"] = PUPIL_DATA_RESAMPLED         

            print(f"===================================================================")
            print(f"FMRI_TARGET_RESAMPLED {FMRI_TARGET_RESAMPLED.shape}")
            print(f"PUPIL_DATA_RESAMPLED {PUPIL_DATA_RESAMPLED.shape}")
            print(f"===================================================================")

            # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

s['data_fmri'] = data_fmri
s['data_eyetracking'] = data_eyetracking
#s['data_respiration'] = data_respiration

print("saving...") # added last 2
file_str = f"group_data_natview_data_fmri_eyetracking"
filename = os.path.join(out_dir,f"{file_str}.npy")
np.save(filename, s)
print("...done")

print(len(data_fmri))

#%% =======================================================================================
