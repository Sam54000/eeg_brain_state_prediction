#%% =======================================================================================
#%load_ext autoreload
#%autoreload 2

import os
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.stats import zscore

import matplotlib.pyplot as plt

import s00_helper_functions_data_bva_edf3 as hf
import importlib
importlib.reload(hf)

#%% =======================================================================================

base_dir = '/data2/Projects/knenning/natview_eeg/'

fmri_data_dir = '/data2/Projects/eeg_fmri_natview/data/fmriprep/derivatives'
#eeg_data_dir = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF'
eeg_proc_data_dir = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES/eeg_features_extraction'
eyetrack_data_dir = '/home/atambini/natview/eyetracking_resampled'
respiration_data_dir = '/home/thoppe/physio-analysis/resp-analysis/resp_stdevs'

#%% =======================================================================================

predHz = 3.8

#2.1/8 = 0.2625

base_out_dir = os.path.join(base_dir,'data_prep', 'prediction_model_data_eeg_features_v2')

out_dir = os.path.join(base_out_dir, f"group_data_Hz-{predHz}")
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

dict_out_dir = os.path.join(base_out_dir, f"dictionary_group_data_Hz-{predHz}")
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

bold_tr = 2.1
pd_tr = 1.05

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
            
            has_fmri, has_eeg, has_pd, has_resp = hf.check_data_exists(sub, ses, task, fmri_data_dir, eeg_proc_data_dir, eyetrack_data_dir, respiration_data_dir)
            bs_exists, bstate_data = hf.get_brainstate_data_all(sub, ses, task, fmri_data_dir, bold_tr)
            
            if not ( bs_exists ):
                print("-- no brainstates")
                continue

            #if not ( has_eeg & has_fmri & has_pd & has_resp ):
            #    print("---> not all features <---")
            #    continue
            
            # ----------------------------------------------------------------------------
            # FMRI DATA

            FMRI_TARGET_RESAMPLED = []
            if bs_exists:
                FMRI_TARGET = bstate_data
                fmri_time = bstate_data['time'].to_numpy()
                # if a different samling rate will be used
                #n_samples = np.round(bstate_data.shape[0] / ((1/bold_tr)/predHz)).astype('int') # tr 2.1sec to 1Hz - 1sec
                #fmri_time_resampled = np.linspace(fmri_time[0], fmri_time[-1], num=n_samples)
                inc = 0.2625
                fmri_time_resampled = np.arange(fmri_time[0], fmri_time[-1]+inc, inc)

                bstate_data = bstate_data.fillna(0)
                FMRI_TARGET_RESAMPLED = pd.DataFrame()
                for col in bstate_data.columns:
                    tmp_spline = CubicSpline(fmri_time, bstate_data[col])
                    FMRI_TARGET_RESAMPLED[col] = tmp_spline(fmri_time_resampled)
                
            # ----------------------------------------------------------------------------
            # EYETRACKING DATA

            PUPIL_DATA_RESAMPLED = []
            if has_pd:
                eyetrack_data_file = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv")
                pd_df = pd.read_csv(eyetrack_data_file, sep='\t', index_col=0)

                pd_time = pd_df['time'].to_numpy()
                pd_spline = CubicSpline(pd_time, pd_df['pupil_size'])
                pd_time_resampled = fmri_time_resampled
                pupil_size_resampled = pd_spline(pd_time_resampled)

                pd_X_spline = CubicSpline(pd_time, pd_df['X_position'])
                pd_X_position_resampled = pd_X_spline(pd_time_resampled)
                pd_Y_spline = CubicSpline(pd_time, pd_df['Y_position'])
                pd_Y_position_resampled = pd_Y_spline(pd_time_resampled)

                pd_mask_spline = CubicSpline(pd_time, pd_df['tmask'])
                pd_tmask_resampled = pd_mask_spline(pd_time_resampled)
                
                PUPIL_DATA_RESAMPLED = pd.DataFrame(
                    data = np.vstack((pd_time_resampled, pupil_size_resampled, pd_X_position_resampled, pd_Y_position_resampled, pd_tmask_resampled)).T,
                    columns = ['time', 'pupil_size', 'X_position', 'Y_position', 'tmask'] )
               
            # ----------------------------------------------------------------------------
            # RESPIRATION DATA

            RESPIRATION_DATA_RESAMPLED = []
            if has_resp:
                if (task[:2]=='tp'):
                    bstask = task
                else:
                    bstask = task[:(len(task)-7)]
                respiration_data_file = os.path.join(respiration_data_dir, f"sub-{sub}_ses-{ses}_task-{bstask}_resp_stdevs.csv")
                resp_df = pd.read_csv(respiration_data_file, sep=',', index_col=0)

                resp_time = resp_df['Time'].to_numpy()
                resp_spline = CubicSpline(resp_time, resp_df['StdDev_6s'])
                resp_time_resampled = fmri_time_resampled
                respiration_resampled = resp_spline(resp_time_resampled)
                RESPIRATION_DATA_RESAMPLED = pd.DataFrame(
                    data = np.vstack((resp_time_resampled, respiration_resampled)).T,
                    columns = ['time', 'respiration'] )
                
            # ----------------------------------------------------------------------------
            # EEG DATA

            sub_dir_eeg = os.path.join(eeg_proc_data_dir, f"sub-{sub}", f"ses-{ses}", "eeg")
            DATA_EEGbandsEnvelopes  = []
            DATA_CustomEnvelopes  = []
            DATA_MorletTFR  = []
            if has_eeg:
                if (task[:2]=='tp'):
                    bstask = task
                else:
                    bstask = task[:(len(task)-7)]

                #sub_dir_eeg = os.path.join(eeg_proc_data_dir, f"sub-{sub}", f"ses-{ses}", "eeg")
                EEGbandsEnvelopes_data_file = os.path.join(sub_dir_eeg, f"sub-{sub}_ses-{ses}_task-{task}_desc-EEGbandsEnvelopes_eeg.pkl")
                EEGbandsEnvelopes_data = np.load(EEGbandsEnvelopes_data_file, allow_pickle=True)
                EEGbandsEnvelopes_data.keys()
                print(f"{EEGbandsEnvelopes_data['feature'].shape}")
                print(f"{EEGbandsEnvelopes_data['times'].shape}")
                #plt.imshow(EEGbandsEnvelopes_data['feature'][1,:,:].T, aspect='auto', vmax=1e-3)
                EEGbandsEnvelopes_spline = CubicSpline(EEGbandsEnvelopes_data['times'], EEGbandsEnvelopes_data['feature'], axis=1)
                EEGbandsEnvelopes_resampled = EEGbandsEnvelopes_spline(fmri_time_resampled)
                #plt.imshow(EEGbandsEnvelopes_resampled[10,:,:].T, aspect='auto', vmax=1e-4)
                
                CustomEnvelopes_data_file = os.path.join(sub_dir_eeg, f"sub-{sub}_ses-{ses}_task-{task}_desc-CustomEnvelopes_eeg.pkl")
                CustomEnvelopes_data = np.load(CustomEnvelopes_data_file, allow_pickle=True)
                CustomEnvelopes_data.keys()
                print(f"{CustomEnvelopes_data['feature'].shape}")
                print(f"{CustomEnvelopes_data['times'].shape}")
                #plt.imshow(CustomEnvelopes_data['feature'][1,:,:].T, aspect='auto', vmax=1e-3)
                CustomEnvelopes_spline = CubicSpline(CustomEnvelopes_data['times'], CustomEnvelopes_data['feature'], axis=1)
                CustomEnvelopes_resampled = CustomEnvelopes_spline(fmri_time_resampled)
               
                MorletTFR_data_file = os.path.join(sub_dir_eeg, f"sub-{sub}_ses-{ses}_task-{task}_desc-MorletTFR_eeg.pkl")
                MorletTFR_data = np.load(MorletTFR_data_file, allow_pickle=True)
                MorletTFR_data.keys()
                print(f"{MorletTFR_data['feature'].shape}")
                print(f"{MorletTFR_data['times'].shape}")
                #plt.imshow(MorletTFR_data['feature'][1,:,:].T, aspect='auto', vmax=1e-3)
                MorletTFR_spline = CubicSpline(MorletTFR_data['times'], MorletTFR_data['feature'], axis=2)
                MorletTFR_resampled = MorletTFR_spline(fmri_time_resampled)
                MorletTFR_resampled = np.moveaxis(MorletTFR_resampled, 2, 1)
                
            # ----------------------------------------------------------------------------

            print(f"fmri_time_resampled {fmri_time_resampled.shape}")
            print(f"FMRI_TARGET_RESAMPLED {FMRI_TARGET_RESAMPLED.shape}")
            print(f"PUPIL_DATA_RESAMPLED {PUPIL_DATA_RESAMPLED.shape}")
            print(f"RESPIRATION_DATA_RESAMPLED {RESPIRATION_DATA_RESAMPLED.shape}")
            print(f"EEGbandsEnvelopes_resampled {EEGbandsEnvelopes_resampled.shape}")
            print(f"CustomEnvelopes_resampled {CustomEnvelopes_resampled.shape}")
            print(f"MorletTFR_resampled {MorletTFR_resampled.shape}")
            print(f"cropping it ...")
            #min_length = np.min( [ len(FMRI_TARGET_RESAMPLED), len(PUPIL_DATA_RESAMPLED), len(RESPIRATION_DATA_RESAMPLED) ] )
            max_time = np.min( [ fmri_time[-1], pd_time[-1], resp_time[-1], EEGbandsEnvelopes_data['times'][-1], CustomEnvelopes_data['times'][-1], MorletTFR_data['times'][-1] ] )
            min_length = np.where((FMRI_TARGET_RESAMPLED['time'] < max_time) == 0)[0][0]

            FMRI_TARGET_RESAMPLED = FMRI_TARGET_RESAMPLED.iloc[:min_length, :]
            PUPIL_DATA_RESAMPLED = PUPIL_DATA_RESAMPLED.iloc[:min_length, :]
            RESPIRATION_DATA_RESAMPLED = RESPIRATION_DATA_RESAMPLED.iloc[:min_length, :]

            fmri_time_resampled = fmri_time_resampled[:min_length]
            EEGbandsEnvelopes_resampled = EEGbandsEnvelopes_resampled[:, :min_length, :]
            CustomEnvelopes_resampled = CustomEnvelopes_resampled[:, :min_length, :]
            MorletTFR_resampled = MorletTFR_resampled[:, :min_length, :]

            DATA_EEGbandsEnvelopes = {
                'data': EEGbandsEnvelopes_resampled,
                'frequencies': EEGbandsEnvelopes_data['frequencies'],
                'channel_info': EEGbandsEnvelopes_data['channels_info'],
                'time': fmri_time_resampled
                }

            DATA_CustomEnvelopes = {
                'data': CustomEnvelopes_resampled,
                'frequencies': CustomEnvelopes_data['frequencies'],
                'channel_info': CustomEnvelopes_data['channels_info'],
                'time': fmri_time_resampled
                }

            DATA_MorletTFR = {
                    'data': MorletTFR_resampled,
                    'frequencies': MorletTFR_data['frequencies'],
                    'channel_info': MorletTFR_data['channels_info'],
                    'time': fmri_time_resampled
                    }

            # # add to one dictionary
            # data_fmri[f"sub-{sub}_ses-{ses}_task-{task}"] = FMRI_TARGET_RESAMPLED
            # data_eyetracking[f"sub-{sub}_ses-{ses}_task-{task}"] = PUPIL_DATA_RESAMPLED 
            # data_respiration[f"sub-{sub}_ses-{ses}_task-{task}"] = RESPIRATION_DATA_RESAMPLED     
            # data_eeg_bands_envelope[f"sub-{sub}_ses-{ses}_task-{task}"] = (EEGbandsEnvelopes_resampled, fmri_time_resampled, EEGbandsEnvelopes_data['frequencies'], EEGbandsEnvelopes_data['channels_info'])
            # data_eeg_custom_envelope[f"sub-{sub}_ses-{ses}_task-{task}"] = (CustomEnvelopes_resampled, fmri_time_resampled, CustomEnvelopes_data['frequencies'], CustomEnvelopes_data['channels_info'])
            # data_eeg_morlet_tfr[f"sub-{sub}_ses-{ses}_task-{task}"]   = (MorletTFR_resampled, fmri_time_resampled, MorletTFR_data['frequencies'], MorletTFR_data['channels_info'])

            # save individual files
            FMRI_TARGET_RESAMPLED.to_csv(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_fmri-brainstates-networks.csv"), index=False)
            PUPIL_DATA_RESAMPLED.to_csv(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.csv"), index=False)
            RESPIRATION_DATA_RESAMPLED.to_csv(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_respiration-variation.csv"), index=False)
            np.save(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-eegbands-envelopes.npy"), DATA_EEGbandsEnvelopes)
            np.save(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-custom-envelopes.npy"), DATA_CustomEnvelopes)
            np.save(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-morlet-tfr.npy"), DATA_MorletTFR)

            print(f"===================================================================")
            print(f"fmri_time_resampled {fmri_time_resampled.shape}")
            print(f"FMRI_TARGET_RESAMPLED {FMRI_TARGET_RESAMPLED.shape}")
            print(f"PUPIL_DATA_RESAMPLED {PUPIL_DATA_RESAMPLED.shape}")
            print(f"RESPIRATION_DATA_RESAMPLED {RESPIRATION_DATA_RESAMPLED.shape}")
            print(f"EEGbandsEnvelopes_resampled {EEGbandsEnvelopes_resampled.shape}")
            print(f"CustomEnvelopes_resampled {CustomEnvelopes_resampled.shape}")
            print(f"MorletTFR_resampled {MorletTFR_resampled.shape}")
            print(f"===================================================================")

            # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# s['data_fmri'] = data_fmri
# s['data_eyetracking'] = data_eyetracking
# s['data_respiration'] = data_respiration
# s['data_eeg_bands_envelope'] = data_eeg_bands_envelope
# s['data_eeg_custom_envelope'] = data_eeg_custom_envelope
# s['data_eeg_morlet_tfr'] = data_eeg_morlet_tfr

# print("saving...") # added last 2
# file_str = f"group_data_natview_data_fmri_eyetracking_respiration"
# filename = os.path.join(dict_out_dir,f"{file_str}.npy")
# np.save(filename, s)
# print("...done")

# print(len(data_fmri))

#%% =======================================================================================
