# %% =================================================================================================
import os, mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
#from mne_connectivity import spectral_connectivity_epochs

# %% =================================================================================================

base_dir = '/data2/Projects/knenning/natview_eeg/playground'
out_dir = os.path.join(base_dir,'eegfmri_datashare_bv_edf2')
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

fmri_data_dir = '/data2/Projects/eeg_fmri_natview/data/fmriprep/derivatives'
eeg_data_dir = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF' #
eyetrack_data_dir = '/home/atambini/natview/eyetracking_resampled'

bold_tr = 2.1

# %% =================================================================================================

sub = '01'
ses = '01'
task = 'checker_run-01' # checker_run-01 rest_run-01 tp_run-01 tp_run-02

sub_list = [ '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
ses_list = [ '01', '02' ]
task_list = [ 'checker_run-01', 'rest_run-01' ] # , 'tp_run-01', 'tp_run-02'

df_overview = pd.DataFrame(columns=['sub', 'ses', 'task', 'n_eeg_tr', 'n_channels', 'n_frequencies', 'n_eeg_samples', 'raw_shape', 'n_brainstates'])
df_overview_idx = -1

for sub in sub_list:
    for task in task_list:
        for ses in ses_list:
            print(f"{sub} - {task}")
            # %% =================================================================================================
            # SAVE BRAINSTATES

            df_overview_idx = df_overview_idx + 1
            df_overview.at[df_overview_idx, 'sub'] = sub
            df_overview.at[df_overview_idx, 'ses'] = ses
            df_overview.at[df_overview_idx, 'task'] = task

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

            all_brainstates_extists = os.path.exists(caps_ts_file) & os.path.exists(caps_pca_file) & os.path.exists(net_yeo7_file) & os.path.exists(net_yeo17_file) & os.path.exists(global_signal_file)
            all_brainstates_extists

            brainstates_df = pd.DataFrame()
            if all_brainstates_extists:
                #brainstates_df = pd.DataFrame()
                # kmeans CAPs
                caps_ts_data = pd.read_csv(caps_ts_file, sep='\t')
                #caps_ts_data = caps_ts_data.iloc[:,:8]
                brainstates_df = pd.concat((brainstates_df,caps_ts_data.add_prefix("ts", axis=1)), axis=1, ignore_index=False, sort=False)
                # PCA CAPs
                caps_pca_data = pd.read_csv(caps_pca_file, sep='\t')
                caps_pca_data = caps_pca_data.iloc[:,:10]
                brainstates_df = pd.concat((brainstates_df,caps_pca_data.add_prefix("pca", axis=1)), axis=1, ignore_index=False, sort=False)
                # Yeo 7 Networks
                net_yeo7_data = pd.read_csv(net_yeo7_file, sep='\t', header=None)
                net_yeo7_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo7_data.columns]
                brainstates_df = pd.concat((brainstates_df,net_yeo7_data.add_prefix("yeo7", axis=1)), axis=1, ignore_index=False, sort=False)
                # Yeo 17 Networks
                net_yeo17_data = pd.read_csv(net_yeo17_file, sep='\t', header=None)
                net_yeo17_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo17_data.columns]
                brainstates_df = pd.concat((brainstates_df,net_yeo17_data.add_prefix("yeo17", axis=1)), axis=1, ignore_index=False, sort=False)
                # Yeo 17 Networks
                global_signal_data = pd.read_csv(global_signal_file, sep='\t', header=None)
                global_signal_data.columns = ["GS"]
                brainstates_df = pd.concat((brainstates_df,global_signal_data), axis=1, ignore_index=False, sort=False)

                n_trs = brainstates_df.shape[0]
                n_trs

                df = pd.DataFrame()
                fmri_time = np.arange(bold_tr/2,n_trs*bold_tr,bold_tr)
                df['time'] = np.around(fmri_time, decimals=2)
                brainstates_df = pd.concat((df,brainstates_df), axis=1, ignore_index=False, sort=False)
                brainstates_df.to_csv(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_fmri-brainstates-networks.csv"), index=False)
                brainstates_df
            

            df_overview.at[df_overview_idx, 'n_brainstates'] = len(brainstates_df)

            # %% =================================================================================================
            # EYETRACKING DATA

            pdtask = task
            eyetrack_data_file = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{pdtask}_eyelink-pupil-eye-position.tsv")

            try:
                pd_df = pd.read_csv(eyetrack_data_file, sep='\t', index_col=0)
                eyetrack_data = pd_df

                eyetrack_data.to_csv(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.csv"), index=False)
                eyetrack_data
            except:
                print("no pd data")

            # %% =================================================================================================
            # EEG DATA - time frequency decompostion

            #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg_{eeg_preproc_step}.set")
                
            eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")

            if (os.path.exists(eeg_data) == False):
                eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")
            
            if (os.path.exists(eeg_data) == False):
                print("no eeg data?")
                continue
            
            try: 
                raw_eeg = mne.io.read_raw_edf(eeg_data, preload=True)
            except:
                print("-------------------------------------------------------------------------------------")
                print("- BROKEN EEG DATA? ------------------------------------------------------------------")
                print("-------------------------------------------------------------------------------------")
                continue

            channel_names = (raw_eeg.ch_names)

            events, event_id = mne.events_from_annotations(raw_eeg)
            #The first column contains the event time in samples. The third column contains the event id.
            scanner_tr_id = event_id['R128']
            print(f"events.shape {events.shape}, #R128(tr) {np.sum(events[:][:,2]==scanner_tr_id)}")

            sfreq = raw_eeg.info['sfreq']
            # split in epochs per tr
            spadding = 1
            event_id, tmin, tmax = scanner_tr_id, 0-spadding, (bold_tr-(1/sfreq))+spadding # remove 1 sample, since bold_tr is new one
            #event_id, tmin, tmax = scanner_tr_id, 0, (bold_tr-(1/sfreq)) # remove 1 sample, since bold_tr is new one
            epochs = mne.Epochs(
                    raw_eeg,
                    events,
                    event_id,
                    tmin,
                    tmax,
                    picks=channel_names,
                    baseline=None,
                    preload=True,
                    decim=1, 
                    event_repeated='merge'
                )

            # define frequencies of interest
            freqs = np.arange(1, 41, 1)
            len(freqs)

            desired_sfreq = 10
            ndecim = np.round(sfreq / desired_sfreq).astype(int)
            # run the TFR decomposition
            # n_cycles= freqs / 2.0,
            tfr_epochs = tfr_morlet(
                epochs,
                freqs,
                n_cycles=freqs / 2.0,
                decim=ndecim,
                average=False,
                return_itc=False,
                n_jobs=None,
            )

            tfr_epochs = tfr_epochs.crop(tmin=0, tmax=bold_tr, include_tmax=False)
            epochs_power = tfr_epochs.data
            print(epochs_power.shape)

    
            scanner_tr_events = events[events[:][:,2]==scanner_tr_id]
            eeg_start = scanner_tr_events[0,0]
            eeg_start_sec = scanner_tr_events[0,0]/sfreq
            eeg_end = scanner_tr_events[scanner_tr_events.shape[0]-1,0]+int(bold_tr/(1/sfreq))
            eeg_end_sec = eeg_end/sfreq

            print(f"{(eeg_end-eeg_start)/ndecim} - {epochs_power.shape[0]*epochs_power.shape[3]}") # double check

            s = {
                'data': epochs_power,
                'freqs': freqs,
                'ch_names': channel_names
                }
            np.save(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-data-tfd.npy"), s)


            df_overview.at[df_overview_idx, 'n_eeg_tr'] = epochs_power.shape[0]
            df_overview.at[df_overview_idx, 'n_channels'] = epochs_power.shape[1]
            df_overview.at[df_overview_idx, 'n_frequencies'] = epochs_power.shape[2]
            df_overview.at[df_overview_idx, 'n_eeg_samples'] = epochs_power.shape[3]
          

            # %% =================================================================================================
            # EEG DATA - BROADBAND AND CONNECTIVITY

            #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.set")
            raw_eeg = mne.io.read_raw_edf(eeg_data, preload=True)
            channel_names = (raw_eeg.ch_names)

            events, event_id = mne.events_from_annotations(raw_eeg)
            #The first column contains the event time in samples. The third column contains the event id.
            scanner_tr_id = event_id['R128']
            print(f"events.shape {events.shape}, #R128(tr) {np.sum(events[:][:,2]==scanner_tr_id)}")

            sfreq = raw_eeg.info['sfreq']    
            #filtered_eeg = mne.filter.filter_data(raw_eeg, sfreq, 0, 60)
            filtered_eeg = raw_eeg.filter(0, 40)
            # split in epochs per tr
            event_id, tmin, tmax = scanner_tr_id, 0, (bold_tr-(1/sfreq)) # remove 1 sample, since bold_tr is new one
            desired_sfreq = 10
            ndecim = np.round(sfreq / desired_sfreq).astype(int)
            epochs = mne.Epochs(
                    filtered_eeg,
                    events,
                    event_id,
                    tmin,
                    tmax,
                    picks=channel_names,
                    baseline=None,
                    preload=True,
                    decim=ndecim, 
                    event_repeated='merge'
                )

            epochs_power = epochs.get_data()
            print(epochs_power.shape)

            scanner_tr_events = events[events[:][:,2]==scanner_tr_id]
            eeg_start = scanner_tr_events[0,0]
            eeg_start_sec = scanner_tr_events[0,0]/sfreq
            eeg_end = scanner_tr_events[scanner_tr_events.shape[0]-1,0]+int(bold_tr/(1/sfreq))
            eeg_end_sec = eeg_end/sfreq

            print(f"{(eeg_end-eeg_start)/ndecim} - {epochs_power.shape[0]*epochs_power.shape[2]}") # double check

            # 'freqs': freqs,
            s = {
                'data': epochs_power,
                'ch_names': channel_names
                }
            np.save(os.path.join(out_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-data-raw.npy"), s)


            df_overview.at[df_overview_idx, 'raw_shape'] = epochs_power.shape
            #print(df_overview)
            # %% =================================================================================================

df_overview.to_csv(os.path.join(out_dir, '_overview.csv'))
print(df_overview)

# %% =================================================================================================