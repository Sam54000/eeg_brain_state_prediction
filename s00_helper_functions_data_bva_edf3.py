import os, nilearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from nilearn import image
import scipy.signal as signal
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.image import concat_imgs, mean_img
from nilearn.plotting import plot_stat_map
import mne
from mne.time_frequency import tfr_morlet, tfr_multitaper
from scipy.interpolate import CubicSpline

# ==============================================================================================================================
# ==============================================================================================================================
def get_brainstate_data(sub, ses, task, brainstate_dir):
    #print("checking if files exist")
    brainstate_data_file = os.path.join(brainstate_dir, f"sub-{sub}_ses-{ses}_task-{task}.txt")

    if ( os.path.exists(brainstate_data_file) ):
        brainstate_data = pd.read_csv(brainstate_data_file, sep='\t') #, index_col=0
    else:
        brainstate_data = np.nan

    return ( os.path.exists(brainstate_data_file) ), brainstate_data


def get_brainstate_data_all(sub, ses, task, fmri_data_dir, bold_tr):
    #print("checking if files exist")

    if (task[:2]=='tp'):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]

    caps_ts_file = os.path.join(fmri_data_dir, 'cap_ts', f"sub-{sub}_ses-{ses}_task-{bstask}.txt")
    caps_pca_file = os.path.join(fmri_data_dir, 'pca_cap_ts', f"sub-{sub}_ses-{ses}_task-{bstask}.txt")
    fmri_timeseries_dir = os.path.join(fmri_data_dir, 'extracted_ts')
    net_yeo7_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_NR_Yeo7.csv")
    net_yeo17_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_NR_Yeo17.csv")
    global_signal_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_NR_GS.csv")
    global_signal_raw_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_GS-raw.csv")

    print(f"caps_ts_file exists - {os.path.exists(caps_ts_file)}")
    print(f"caps_pca_file exists - {os.path.exists(caps_pca_file)}")
    print(f"net_yeo7_file exists - {os.path.exists(net_yeo7_file)}")
    print(f"net_yeo17_file exists - {os.path.exists(net_yeo17_file)}")
    print(f"global_signal_file exists - {os.path.exists(global_signal_file)}")
    print(f"global_signal_raw_file exists - {os.path.exists(global_signal_raw_file)}")

    #  & os.path.exists(caps_pca_file)
    all_brainstates_extists = os.path.exists(caps_ts_file) & os.path.exists(net_yeo7_file) & os.path.exists(net_yeo17_file) & os.path.exists(global_signal_file) & os.path.exists(global_signal_raw_file)
    all_brainstates_extists

    brainstates_df = pd.DataFrame()
    if all_brainstates_extists:
        # kmeans CAPs
        caps_ts_data = pd.read_csv(caps_ts_file, sep='\t')
        #caps_ts_data = caps_ts_data.iloc[:,:8]
        brainstates_df = pd.concat((brainstates_df,caps_ts_data.add_prefix("ts", axis=1)), axis=1, ignore_index=False, sort=False)
        # PCA CAPs
        #caps_pca_data = pd.read_csv(caps_pca_file, sep='\t')
        #caps_pca_data = caps_pca_data.iloc[:,:10]
        #brainstates_df = pd.concat((brainstates_df,caps_pca_data.add_prefix("pca", axis=1)), axis=1, ignore_index=False, sort=False)
        # Yeo 7 Networks
        net_yeo7_data = pd.read_csv(net_yeo7_file, sep='\t', header=None)
        net_yeo7_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo7_data.columns]
        brainstates_df = pd.concat((brainstates_df,net_yeo7_data.add_prefix("yeo7", axis=1)), axis=1, ignore_index=False, sort=False)
        # Yeo 17 Networks
        net_yeo17_data = pd.read_csv(net_yeo17_file, sep='\t', header=None)
        net_yeo17_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo17_data.columns]
        brainstates_df = pd.concat((brainstates_df,net_yeo17_data.add_prefix("yeo17", axis=1)), axis=1, ignore_index=False, sort=False)
        # GS
        global_signal_data = pd.read_csv(global_signal_file, sep='\t', header=None)
        global_signal_data.columns = ["GS"]
        brainstates_df = pd.concat((brainstates_df,global_signal_data), axis=1, ignore_index=False, sort=False)
        # GS-raw
        global_signal_raw_data = pd.read_csv(global_signal_raw_file, sep='\t', header=None)
        global_signal_raw_data.columns = ["GS_raw"]
        brainstates_df = pd.concat((brainstates_df,global_signal_raw_data), axis=1, ignore_index=False, sort=False)

        n_trs = brainstates_df.shape[0]
        n_trs

        df = pd.DataFrame()
        fmri_time = np.arange(bold_tr/2,n_trs*bold_tr,bold_tr)
        df['time'] = np.around(fmri_time, decimals=2)
        brainstates_df = pd.concat((df,brainstates_df), axis=1, ignore_index=False, sort=False)
    else:
        brainstates_df = np.nan

    brainstate_data = brainstates_df
    return all_brainstates_extists, brainstate_data

           

# ==============================================================================================================================
# ==============================================================================================================================
def check_data_exists(sub, ses, task, fmri_data_dir, eeg_proc_data_dir, eyetrack_data_dir, respiration_data_dir):
    #print("checking if files exist")
    sub_dir = os.path.join(fmri_data_dir, f"sub-{sub}", f"ses-{ses}")

    if (task[:2]=='tp'):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]
    
    # sub-01_ses-01_task-checker_space-T1w_desc-preproc_bold.nii.gz
    fmri_data = os.path.join(sub_dir, "func", f"sub-{sub}_ses-{ses}_task-{bstask}_space-T1w_desc-preproc_bold.nii.gz")
    #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")
    #if (os.path.exists(eeg_data) == False):
    #    eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")
    eeg_data = os.path.join(eeg_proc_data_dir, f"sub-{sub}", f"ses-{ses}", "eeg", f"sub-{sub}_ses-{ses}_task-{task}_desc-EEGbandsEnvelopes_eeg.pkl")
    eyetrack_data = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv")
    respiration_data = os.path.join(respiration_data_dir, f"sub-{sub}_ses-{ses}_task-{bstask}_resp_stdevs.csv")

    #print(eeg_data)
    print(f"fmri data exists - {os.path.exists(fmri_data)}")
    print(f"eeg data exists - {os.path.exists(eeg_data)}")
    print(f"pd data exists - {os.path.exists(eyetrack_data)}")
    print(f"resp data exists - {os.path.exists(respiration_data)}")

    #return ( os.path.exists(fmri_data) & os.path.exists(eeg_data) )
    return os.path.exists(fmri_data), os.path.exists(eeg_data), os.path.exists(eyetrack_data), os.path.exists(respiration_data)

# ==============================================================================================================================
