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

    if (task[:2]!='ch') & (task[:2]!='re') & (task[:2]!='in'):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]

    print(f"-- {bstask}")

    caps_ts_file = os.path.join(fmri_data_dir, 'cap_ts', f"sub-{sub}_ses-{ses}_task-{bstask}.txt")
    #caps_pca_file = os.path.join(fmri_data_dir, 'pca_cap_ts', f"sub-{sub}_ses-{ses}_task-{bstask}.txt")
    fmri_timeseries_dir = os.path.join(fmri_data_dir, 'extracted_ts')
    net_yeo7_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_NR_Yeo7.csv")
    net_yeo17_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_NR_Yeo17.csv")
    global_signal_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_NR_GS.csv")
    #global_signal_raw_file = os.path.join(fmri_timeseries_dir , f"sub-{sub}_ses-{ses}_task-{bstask}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_GS-raw.csv")

    print(f"caps_ts_file exists - {os.path.exists(caps_ts_file)}")
    #print(f"caps_pca_file exists - {os.path.exists(caps_pca_file)}")
    print(f"net_yeo7_file exists - {os.path.exists(net_yeo7_file)}")
    print(f"net_yeo17_file exists - {os.path.exists(net_yeo17_file)}")
    print(f"global_signal_file exists - {os.path.exists(global_signal_file)}")
    #print(f"global_signal_raw_file exists - {os.path.exists(global_signal_raw_file)}")

    #  & os.path.exists(caps_pca_file)
    #all_brainstates_extists = os.path.exists(caps_ts_file) & os.path.exists(net_yeo7_file) & os.path.exists(net_yeo17_file) & os.path.exists(global_signal_file) & os.path.exists(global_signal_raw_file)
    all_brainstates_extists = os.path.exists(caps_ts_file) & os.path.exists(net_yeo7_file) & os.path.exists(net_yeo17_file) & os.path.exists(global_signal_file)
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
        ## GS-raw
        #global_signal_raw_data = pd.read_csv(global_signal_raw_file, sep='\t', header=None)
        #global_signal_raw_data.columns = ["GS_raw"]
        #brainstates_df = pd.concat((brainstates_df,global_signal_raw_data), axis=1, ignore_index=False, sort=False)

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
def check_data_exists(sub, ses, task, fmri_data_dir, eeg_data_dir, eyetrack_data_dir, respiration_data_dir):
    #print("checking if files exist")
    sub_dir = os.path.join(fmri_data_dir, f"sub-{sub}", f"ses-{ses}")

    if (task[:2]!='ch') & (task[:2]!='re') & (task[:2]!='in'):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]
    
    # sub-01_ses-01_task-checker_space-T1w_desc-preproc_bold.nii.gz
    fmri_data = os.path.join(sub_dir, "func", f"sub-{sub}_ses-{ses}_task-{bstask}_space-T1w_desc-preproc_bold.nii.gz")
    eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")
    if (os.path.exists(eeg_data) == False):
        eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")

    eyetrack_data = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv")
    respiration_data = os.path.join(respiration_data_dir, f"sub-{sub}_ses-{ses}_task-{bstask}_resp_stdevs.csv")

    #print(eeg_data)
    print(f"fmri data exists - {os.path.exists(fmri_data)}")
    #print(f"eeg data exists - {os.path.exists(eeg_data)}")
    print(f"pd data exists - {os.path.exists(eyetrack_data)}")
    print(f"resp data exists - {os.path.exists(respiration_data)}")

    #return ( os.path.exists(fmri_data) & os.path.exists(eeg_data) )
    return os.path.exists(fmri_data), os.path.exists(eeg_data), os.path.exists(eyetrack_data), os.path.exists(respiration_data)

# ==============================================================================================================================
# ==============================================================================================================================
def get_peak_voxel(predHz, sub, ses, task, fmri_data_dir, eyetrack_dir, image_dir):
    # run glm to get roi
    sub_dir = os.path.join(fmri_data_dir, f"sub-{sub}", f"ses-{ses}")
    # sub-01_ses-01_task-checker_space-T1w_desc-preproc_bold.nii.gz
    fmri_data = os.path.join(sub_dir, "func", f"sub-{sub}_ses-{ses}_task-{task}_space-T1w_desc-preproc_bold.nii.gz")
    fmri_data_mask = os.path.join(sub_dir, "func", f"sub-{sub}_ses-{ses}_task-{task}_space-T1w_desc-brain_mask.nii.gz")
    struct_data = os.path.join(sub_dir, "anat", f"sub-{sub}_ses-{ses}_desc-preproc_T1w.nii.gz")

    bold = nilearn.image.load_img(fmri_data)
    n_scans = bold.header['dim'][4]
    bold_tr = bold.header['pixdim'][4]

    confounds_simple, sample_mask = load_confounds(
        fmri_data,
        strategy=["high_pass", "motion"],
        motion="basic", wm_csf="basic", global_signal='derivatives', compcor='anat_combined'
        )

    # these are the corresponding onset times
    onsets = np.arange(20,(n_scans*bold_tr),40)
    duration = 20. * np.ones(len(onsets))
    conditions = ['task'] * len(onsets)
    frame_times = np.arange(n_scans) * bold_tr 

    hrf_model = 'spm'
    events = pd.DataFrame({'trial_type': conditions, 'onset': onsets, 'duration': duration})
    print(events)
    X2 = make_first_level_design_matrix(frame_times, events,
                                    drift_model='polynomial', drift_order=3,
                                    hrf_model=hrf_model,
                                    add_regs=confounds_simple)
    design_matrix = X2
    # plt.figure(); plt.imshow(design_matrix, aspect='auto')
    # plotting.plot_design_matrix(design_matrix)

    fmri_glm = FirstLevelModel()
    #fmri_img_smoothed = image.smooth_img(bold_file, fwhm=6)
    fmri_glm = fmri_glm.fit(fmri_data, design_matrices=design_matrix)

    contrast = {'task': np.hstack(([1],np.zeros(design_matrix.shape[1]-1)))}
    contrast

    ctrst = np.hstack(([1],np.zeros(design_matrix.shape[1]-1)))
    zmap = fmri_glm.compute_contrast(ctrst, output_type='z_score')
    zmap

    #z_image_path = os.path.join(out_dir,f"{sub_id}_{task}_zmaps.nii.gz")
    #zmap.to_filename(z_image_path)
    fmri_img = concat_imgs(fmri_data)
    mean_fmri_img = mean_img(fmri_img)
    fig = plt.figure(figsize=(8,3)) #cut_coords=5,
    plot_stat_map(
        zmap,
        bg_img=mean_fmri_img,
        threshold=3.0,
        display_mode="ortho",
        black_bg=True,
        title="Active vs Rest (Z>3)",
    )
    plot_str = f"sub-{sub}_ses-{ses}_task-{task}_activation_zmap"
    plt.savefig(os.path.join(image_dir,f"{plot_str}.png"), bbox_inches='tight', dpi=300)

    peak_zmap = np.max(image.get_data(zmap).flatten())
    peak_voxel_mask = nilearn.image.math_img(f"(i1 == {peak_zmap})", i1=zmap)
    masker = nilearn.maskers.NiftiMasker(mask_img=peak_voxel_mask) 
    #bold_smoothed = nilearn.image.smooth_img(bold_data, fwhm=12) #12
    bold_ts = masker.fit_transform(fmri_data)
    bold_ts = zscore(bold_ts)

    # get pupil data
    try:
        pupils = pd.read_csv(os.path.join(eyetrack_dir, f"sub-{sub}_ses-{ses}_task-{task}.tsv"), sep='\t')
        pup_data = pupils['PD'].to_numpy()
        pup_data = zscore(resample(pup_data, n_scans, axis=0))
    except:  
        pup_data = np.zeros(n_scans)
        print("no pupilometry")


    target_data = design_matrix["task"].to_numpy()  
    #target_data_norm = (target_data-np.min(target_data))/(np.max(target_data)-np.min(target_data))
    target_data_norm = zscore(target_data)
    fig = plt.figure(figsize=(6,2))
    plt.plot(target_data_norm, color='k', linestyle='--', linewidth=2, zorder=1)
    plt.plot(bold_ts, linewidth=2, zorder=2)
    plt.plot(pup_data, linewidth=1, zorder=3)
    plt.xlabel('Time')
    plt.title(f"sub-{sub}_ses-{ses}_task-{task} peak voxel")
    plot_str = f"sub-{sub}_ses-{ses}_task-{task}_activation_zmap_peak_voxel"
    plt.savefig(os.path.join(image_dir,f"{plot_str}.png"), bbox_inches='tight', dpi=300)

    FMRI_TARGET = bold_ts
    num = np.round(FMRI_TARGET.shape[0] / ((1/bold_tr)/predHz)).astype('int') # tr 2.1sec to 1Hz - 1sec
    FMRI_TARGET_RESAMPLED = resample(FMRI_TARGET, num, axis=0).T
    print(FMRI_TARGET_RESAMPLED.shape)
    # fig = plt.figure(); plt.plot(FMRI_TARGET_RESAMPLED)

    return FMRI_TARGET_RESAMPLED, bold_tr

# ==============================================================================================================================
# ==============================================================================================================================
def get_eeg_features(predHz, sub, ses, task, eeg_data_dir, electrode_labels, bold_tr, iter_freqs):
    # sub-04_ses-01_task-rest_preprocess_eeg_v2.set
    #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_run-01_eeg_{eeg_preproc_step}.set")
    #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")
    eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg.edf")
    if (os.path.exists(eeg_data) == False):
        eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")
    raw_eeg = mne.io.read_raw_edf(eeg_data, preload=True)
    current_sfreq = raw_eeg.info["sfreq"]
    print(f"freqency = {current_sfreq}")
    #print(raw_eeg.__dict__)

    #electrode_labels = raw_eeg.ch_names
    #electrode_labels = ['O1', 'O2', 'Oz']

    # get start time --------------------------------------------------------------------------
    # "R128": "Scanner trigger onset, code appears every TR during simultaneous EEG-fMRI recording",
    # "S1": "Task Start",
    # "S5": "1s trial onset, code appears every 1 second during task/video",
    # "S99": "Task End"

    # The array of events.
    # The first column contains the event time in samples, with first_samp included. 
    # The third column contains the event id.
    events, event_id = mne.events_from_annotations(raw_eeg)

    task_start_id = event_id['S  1']
    task_end_id = event_id['S 99']
    scanner_tr_id = event_id['R128']
    print(f"events.shape {events.shape}, #R128(tr) {np.sum(events[:][:,2]==scanner_tr_id)}")
    scanner_tr_events = events[events[:][:,2]==scanner_tr_id]
    task_start_events = events[events[:][:,2]==task_start_id]
    task_end_events = events[events[:][:,2]==task_end_id]

    sfreq = raw_eeg.info['sfreq']
    #print(sfreq)

    events[-1][0]/sfreq

    eeg_start = scanner_tr_events[0,0]
    eeg_start_sec = scanner_tr_events[0,0]/sfreq

    eeg_end = scanner_tr_events[scanner_tr_events.shape[0]-1,0]+int(bold_tr/(1/sfreq))
    eeg_end_sec = eeg_end/sfreq

    last_tr_events_sec = (scanner_tr_events[scanner_tr_events.shape[0]-1,0]/sfreq) # + bold_tr
    print(f"eeg_start_sec {eeg_start_sec:.2f} - last_tr_events_sec {last_tr_events_sec:.2f}")
    print(f"task_start_events {task_start_events[0,0]/sfreq:.2f} - task_end_events {task_end_events[0,0]/sfreq:.2f}")

    #%% =======================================================================================

    # 30 sec befor and after
    spadding = 1
    event_id, tmin, tmax = scanner_tr_id, 0-spadding, (bold_tr-(1/sfreq))+spadding
    #event_id, tmin, tmax = task_start_id, -30, 630
    baseline = None # (0, 0)

    if (raw_eeg.info['sfreq'] == 5000):
        current_sfreq = raw_eeg.info["sfreq"]
        desired_sfreq = 250  # Hz
        decim = np.round(current_sfreq / desired_sfreq).astype(int)
        obtained_sfreq = current_sfreq / decim
        lowpass_freq = obtained_sfreq / 2.0
        raw_filtered = raw_eeg.copy().filter(l_freq=None, h_freq=lowpass_freq)
    else:
        raw_filtered = raw_eeg.copy()
        decim = 1

    # picks='Oz'
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id,
        tmin,
        tmax,
        picks=electrode_labels,
        baseline=baseline,
        preload=True,
        decim=decim, 
        event_repeated='merge'
    )

    # define frequencies of interest
    inc = 0.3
    freqs = np.arange(0.3, 40+inc, inc)
    len(freqs)

    # run the TFR decomposition
    # n_cycles= freqs / 2.0,
    tfr_epochs = tfr_morlet(
        epochs,
        freqs,
        n_cycles=freqs / 2.0,
        decim=1,
        average=False,
        return_itc=False,
        n_jobs=None,
    )

    tfr_epochs = tfr_epochs.crop(tmin=0, tmax=bold_tr, include_tmax=False) # (bold_tr-(1/sfreq))
    epochs_power = tfr_epochs.data
    print(epochs_power.shape)

    #%% =======================================================================================

    n_elecs = len(electrode_labels)
    combined_electrode_features = np.array([])
    check_stacking = np.array([])
    check_electrodes = np.array([])

    plt.close('all)')
    eidx = 0; elabel = electrode_labels[eidx]
    for eidx, elabel in enumerate(electrode_labels):
        print(f"----------------------------------------------------")
        print(f"electrode {eidx} - {elabel}")
        print(f"----------------------------------------------------")
        # events x elecs x freq x time    
        #print(epochs_power.shape)
        stock = np.array([])
        for ii in np.arange(epochs_power.shape[0]):
            tmp = epochs_power[ii,eidx,:,:]
            stock = np.hstack([stock, tmp]) if stock.size else tmp

        # ------------------------------------------------------------------
        #downsample to 4Hz
        num = np.round(stock.shape[1] / (current_sfreq/predHz)).astype('int') # 250Hz to 4Hz - 0.25sec
        stock_resampled = resample(stock, num, axis=1)
        # ------------------------------------------------------------------
        # avgerage over frequency bands
        #iter_freqs = [("0-4", 0, 4), ("4-6", 4, 6), ("6-10", 6, 10), ("10-13", 10, 13), ("13-18", 13, 18), 
        #            ("18-24", 18, 24), ("24-32", 24, 32), ("32-41", 32, 41), ("41-51", 41, 51), ("51-65", 51, 65)  ]
    
        #iter_freqs = [("6-10", 6, 10), ("10-13", 10, 13), ("13-18", 13, 18)]
    
        freq_indices = freqs 
        n_fbands = len(iter_freqs)
        frequency_map = np.zeros((n_fbands,stock_resampled.shape[1]))
        freq_labels = np.array([])
        for idx, fband in enumerate(iter_freqs):
            #print(fband)
            lt = fband[1]
            ut = fband[2]
            freq_labels = np.hstack([freq_labels, np.array([fband[0]])]) if freq_labels.size else np.array([fband[0]])
            fband_idx = (freq_indices > lt) & (freq_indices <= ut)
            #print(f"{fband} - {np.sum(fband_idx)}")
            frequency_map[idx,:] = np.nanmean(stock_resampled[fband_idx,:], axis=0)
            frequency_map[idx,:] = frequency_map[idx,:]-np.nanmean(frequency_map[idx,:])

        #f = plt.figure(figsize=(6, 6))
        #ax1 = plt.subplot(211)
        #ax1.plot(frequency_map.T)
        #ax1.set_xlim(0,frequency_map.shape[1])
        #plt.xticks(ticks=np.arange(0,900,100),labels=np.arange(0,225,25))
        #ax2 = plt.subplot(212)
        #ax2.imshow(frequency_map, aspect='auto')
        #plt.ylabel('Freqs (Hz)')
        #plt.yticks(ticks=np.arange(len(iter_freqs)),labels=freq_labels)
        #plt.xlabel('Time (sec)')
        #plt.xticks(ticks=np.arange(0,900,100),labels=np.arange(0,225,25))
        #plt.suptitle(f"electrode {eidx} - {elabel} - {eeg_preproc_step}", y=0.925, va='center')
        #plot_str = f"tfa_morlet_elec_{eidx}_{elabel}_{eeg_preproc_step}"
        #plt.savefig(os.path.join(out_dir,f"{plot_str}.png"), dpi=300, bbox_inches="tight")
        ##plt.savefig(os.path.join(out_dir,f"{plot_str}.svg"), dpi=300, bbox_inches="tight")
        
        # ------------------------------------------------------------------
        ## make sure same size
        frequency_features = frequency_map
        #frequency_features = frequency_map[:,:FMRI_TARGET_RESAMPLED.shape[1]]
        #FMRI_TARGET_RESAMPLED = FMRI_TARGET_RESAMPLED[:,:frequency_features.shape[1]]
        #print(f"eeg features {frequency_features.shape}")
        #print(f"target data {FMRI_TARGET_RESAMPLED.shape}")

        #combined_electrode_features = np.vstack([combined_electrode_features, frequency_features]) if combined_electrode_features.size else frequency_features
        combined_electrode_features = np.dstack([combined_electrode_features, frequency_features]) if combined_electrode_features.size else frequency_features
        
        print(f"eeg features combined {combined_electrode_features.shape}")
        ##tmp = np.repeat(np.reshape(np.arange(frequency_map.shape[0]),(-1,1)),2,axis=1)
        #tmp = np.reshape(np.arange(frequency_map.shape[0]),(1,-1))
        #check_stacking = np.hstack([check_stacking, tmp]) if check_stacking.size else tmp
        #check_electrodes = np.hstack([check_electrodes, np.repeat(elabel,6,axis=0)]) if check_electrodes.size else np.repeat(elabel,6,axis=0)

        # ------------------------------------------------------------------

    EEG_FEATURES = combined_electrode_features
    return EEG_FEATURES

# ==============================================================================================================================
# ==============================================================================================================================
def get_eeg_time_frequency_features(predHz, sub, ses, task, eeg_data_dir, electrode_labels, bold_tr, tfr_freqs):
    # sub-04_ses-01_task-rest_preprocess_eeg_v2.set
    #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_run-01_eeg_{eeg_preproc_step}.set")
    eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")
    if (os.path.exists(eeg_data) == False):
        #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")
        eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg.edf")
    print(eeg_data)
    raw_eeg = mne.io.read_raw_edf(eeg_data, preload=True)
    current_sfreq = raw_eeg.info["sfreq"]
    print(f"freqency = {current_sfreq}")
    #print(raw_eeg.__dict__)

    #electrode_labels = raw_eeg.ch_names
    #electrode_labels = ['O1', 'O2', 'Oz']

    # get start time --------------------------------------------------------------------------
    # "R128": "Scanner trigger onset, code appears every TR during simultaneous EEG-fMRI recording",
    # "S1": "Task Start",
    # "S5": "1s trial onset, code appears every 1 second during task/video",
    # "S99": "Task End"

    # The array of events.
    # The first column contains the event time in samples, with first_samp included. 
    # The third column contains the event id.
    events, event_id = mne.events_from_annotations(raw_eeg)

    task_start_id = event_id['S  1']
    task_end_id = event_id['S 99']
    scanner_tr_id = event_id['R128']
    print(f"events.shape {events.shape}, #R128(tr) {np.sum(events[:][:,2]==scanner_tr_id)}")
    scanner_tr_events = events[events[:][:,2]==scanner_tr_id]
    task_start_events = events[events[:][:,2]==task_start_id]
    task_end_events = events[events[:][:,2]==task_end_id]

    sfreq = raw_eeg.info['sfreq']
    #print(sfreq)

    events[-1][0]/sfreq

    eeg_start = scanner_tr_events[0,0]
    eeg_start_sec = scanner_tr_events[0,0]/sfreq

    eeg_end = scanner_tr_events[scanner_tr_events.shape[0]-1,0]+int(bold_tr/(1/sfreq))
    eeg_end_sec = eeg_end/sfreq

    last_tr_events_sec = (scanner_tr_events[scanner_tr_events.shape[0]-1,0]/sfreq) # + bold_tr
    print(f"eeg_start_sec {eeg_start_sec:.2f} - last_tr_events_sec {last_tr_events_sec:.2f}")
    print(f"task_start_events {task_start_events[0,0]/sfreq:.2f} - task_end_events {task_end_events[0,0]/sfreq:.2f}")

    ## %% =======================================================================================

    # add padding for boundary effects
    spadding = 1
    event_id, tmin, tmax = scanner_tr_id, 0-spadding, (bold_tr-(1/sfreq))+spadding
    #event_id, tmin, tmax = task_start_id, -30, 630
    baseline = None # (0, 0)

    if (raw_eeg.info['sfreq'] == 5000):
        current_sfreq = raw_eeg.info["sfreq"]
        desired_sfreq = 250  # Hz
        decim = np.round(current_sfreq / desired_sfreq).astype(int)
        obtained_sfreq = current_sfreq / decim
        lowpass_freq = obtained_sfreq / 2.0
        raw_filtered = raw_eeg.copy().filter(l_freq=None, h_freq=lowpass_freq)
    else:
        raw_filtered = raw_eeg.copy()
        decim = 1

    current_sfreq = raw_eeg.info["sfreq"]
    print(f"freqency = {current_sfreq}")

    # picks='Oz'
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id,
        tmin,
        tmax,
        picks=electrode_labels,
        baseline=baseline,
        preload=True,
        decim=decim, 
        event_repeated='merge'
    )

    # define frequencies of interest
    #inc = 0.3
    #freqs = np.arange(0.3, 40+inc, inc)
    freqs = tfr_freqs
    len(freqs)

    ## run the TFR decomposition
    ## n_cycles= freqs / 2.0,
    #tfr_epochs = tfr_morlet(
    #    epochs,
    #    freqs,
    #    n_cycles=freqs / 2.0,
    #    decim=1,
    #    average=False,
    #    return_itc=False,
    #    n_jobs=None,
    #)

    tfr_epochs = tfr_multitaper(
        epochs,
        freqs = freqs,
        n_cycles = freqs / 2,
        time_bandwidth = 2.0,
        return_itc=False,
        average=False,
        decim=1,
    )

    tfr_epochs = tfr_epochs.crop(tmin=0, tmax=bold_tr, include_tmax=False) # (bold_tr-(1/sfreq))
    epochs_power = tfr_epochs.data
    print(epochs_power.shape)

    # stack and resample to 
    n_elecs = len(electrode_labels)
    EEG_DATA_RESAMPLED = np.array([])
    check_stacking = np.array([])
    check_electrodes = np.array([])
    
    eidx = 0; elabel = electrode_labels[eidx]
    for eidx, elabel in enumerate(electrode_labels):
        print(f"----------------------------------------------------")
        print(f"electrode {eidx} - {elabel}")
        print(f"----------------------------------------------------")
        # events x elecs x freq x time    
        #print(epochs_power.shape)
        stock = np.array([])
        for ntr in np.arange(epochs_power.shape[0]):
            tmp = epochs_power[ntr,eidx,:,:]
            stock = np.hstack([stock, tmp]) if stock.size else tmp

        # ------------------------------------------------------------------
        #downsample to 4Hz
        n_samples = np.round(stock.shape[1] / (current_sfreq/predHz)).astype('int') # 250Hz to 4Hz - 0.25sec

        eeg_time = np.arange(0, stock.shape[1]) * 1/current_sfreq
        eeg_spl = CubicSpline(eeg_time, stock.T)
        eeg_time_resampled = np.linspace(1.05, eeg_time[-1], num=n_samples)
        stock_resampled = eeg_spl(eeg_time_resampled)
        EEG_DATA_RESAMPLED = np.dstack([EEG_DATA_RESAMPLED, stock_resampled]) if EEG_DATA_RESAMPLED.size else stock_resampled

   
    print(f"EEG_DATA_RESAMPLED {EEG_DATA_RESAMPLED.shape}")

    # ------------------------------------------------------------------

    EEG_FEATURES = EEG_DATA_RESAMPLED
    return EEG_FEATURES, eeg_time_resampled

# ==============================================================================================================================
# ==============================================================================================================================
def get_eeg_raw_features(predHz, sub, ses, task, eeg_data_dir, electrode_labels, bold_tr):
    # sub-04_ses-01_task-rest_preprocess_eeg_v2.set
    #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_run-01_eeg_{eeg_preproc_step}.set")
    eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")
    if (os.path.exists(eeg_data) == False):
        eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")
    raw_eeg = mne.io.read_raw_edf(eeg_data, preload=True)
    current_sfreq = raw_eeg.info["sfreq"]
    print(f"freqency = {current_sfreq}")
    #print(raw_eeg.__dict__)

    #electrode_labels = raw_eeg.ch_names
    #electrode_labels = ['O1', 'O2', 'Oz']

    # get start time --------------------------------------------------------------------------
    # "R128": "Scanner trigger onset, code appears every TR during simultaneous EEG-fMRI recording",
    # "S1": "Task Start",
    # "S5": "1s trial onset, code appears every 1 second during task/video",
    # "S99": "Task End"

    # The array of events.
    # The first column contains the event time in samples, with first_samp included. 
    # The third column contains the event id.
    events, event_id = mne.events_from_annotations(raw_eeg)

    task_start_id = event_id['S  1']
    task_end_id = event_id['S 99']
    scanner_tr_id = event_id['R128']
    print(f"events.shape {events.shape}, #R128(tr) {np.sum(events[:][:,2]==scanner_tr_id)}")
    scanner_tr_events = events[events[:][:,2]==scanner_tr_id]
    task_start_events = events[events[:][:,2]==task_start_id]
    task_end_events = events[events[:][:,2]==task_end_id]

    sfreq = raw_eeg.info['sfreq']
    #print(sfreq)

    events[-1][0]/sfreq

    eeg_start = scanner_tr_events[0,0]
    eeg_start_sec = scanner_tr_events[0,0]/sfreq

    eeg_end = scanner_tr_events[scanner_tr_events.shape[0]-1,0]+int(bold_tr/(1/sfreq))
    eeg_end_sec = eeg_end/sfreq

    last_tr_events_sec = (scanner_tr_events[scanner_tr_events.shape[0]-1,0]/sfreq) # + bold_tr
    print(f"eeg_start_sec {eeg_start_sec:.2f} - last_tr_events_sec {last_tr_events_sec:.2f}")
    print(f"task_start_events {task_start_events[0,0]/sfreq:.2f} - task_end_events {task_end_events[0,0]/sfreq:.2f}")

    ## %% =======================================================================================

    # add padding for boundary effects
    spadding = 1
    event_id, tmin, tmax = scanner_tr_id, 0-spadding, (bold_tr-(1/sfreq))+spadding
    #event_id, tmin, tmax = task_start_id, -30, 630
    baseline = None # (0, 0)

    if (raw_eeg.info['sfreq'] == 5000):
        current_sfreq = raw_eeg.info["sfreq"]
        desired_sfreq = 250  # Hz
        decim = np.round(current_sfreq / desired_sfreq).astype(int)
        obtained_sfreq = current_sfreq / decim
        lowpass_freq = obtained_sfreq / 2.0
        raw_filtered = raw_eeg.copy().filter(l_freq=None, h_freq=lowpass_freq)
    else:
        raw_filtered = raw_eeg.copy()
        decim = 1

    current_sfreq = raw_eeg.info["sfreq"]
    print(f"freqency = {current_sfreq}")


    raw_filtered = raw_filtered.filter(0, 40)
    #raw_filtered = raw_filtered.filter(8, 12)

    # foo = raw_filtered.get_data()
    # foo.shape
    # plt.plot(foo[1,:])

    # picks='Oz'
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id,
        tmin,
        tmax,
        picks=electrode_labels,
        baseline=baseline,
        preload=True,
        decim=decim, 
        event_repeated='merge'
    )


    epochs = epochs.crop(tmin=0, tmax=bold_tr, include_tmax=False) # (bold_tr-(1/sfreq))
    epochs_power = epochs.get_data()
    print(epochs_power.shape)

    # stack and resample to 
    n_elecs = len(electrode_labels)
    EEG_DATA_RESAMPLED = np.array([])
    check_stacking = np.array([])
    check_electrodes = np.array([])
    
    eidx = 0; elabel = electrode_labels[eidx]
    for eidx, elabel in enumerate(electrode_labels):
        print(f"----------------------------------------------------")
        print(f"electrode {eidx} - {elabel}")
        print(f"----------------------------------------------------")
        # events x elecs x freq x time    
        #print(epochs_power.shape)
        stock = np.array([])
        for ntr in np.arange(epochs_power.shape[0]):
            tmp = epochs_power[ntr,eidx,:]
            stock = np.hstack([stock, tmp]) if stock.size else tmp

        # ------------------------------------------------------------------
        #downsample to 4Hz
        n_samples = np.round(stock.shape[0] / (current_sfreq/predHz)).astype('int') # 250Hz to 4Hz - 0.25sec

        eeg_time = np.arange(0, stock.shape[0]) * 1/current_sfreq
        eeg_spl = CubicSpline(eeg_time, stock.T)
        eeg_time_resampled = np.linspace(1.05, eeg_time[-1], num=n_samples) # or start with 1.05 or 0?
        stock_resampled = eeg_spl(eeg_time_resampled)
        EEG_DATA_RESAMPLED = np.vstack([EEG_DATA_RESAMPLED, stock_resampled]) if EEG_DATA_RESAMPLED.size else stock_resampled

   
    EEG_DATA_RESAMPLED = EEG_DATA_RESAMPLED.T
    print(f"EEG_DATA_RESAMPLED {EEG_DATA_RESAMPLED.shape}")

    # ------------------------------------------------------------------

    EEG_FEATURES = EEG_DATA_RESAMPLED
    return EEG_FEATURES, eeg_time_resampled

# ==============================================================================================================================
# ==============================================================================================================================
def get_eeg_alertness_index(predHz, sub, ses, task, eeg_data_dir, bold_tr):
    
    #- EEG-based alertness index
    #- A measure of alertness was computed from the EEG signal (averaged over channels P3, P4, Pz, O1, O2, Oz) within the 2.1 s interval of each fMRI time point (TR) by taking the ratio of the root mean square (rms) amplitude in the 8–12 Hz range over the rms amplitude in the 3–7 Hz range (alpha/theta ratio). Various EEG-based metrics have been associated with wakefulness; however, many are related to the ratio of power in middle frequency bands (i.e., alpha, beta) to the power in lower frequency bands (i.e., delta, theta) (Olbrich et al., 2009; Klimesch, 1999; Oken et al., 2006; Jobert et al., 1994; Wong et al., 2013; Horovitz et al., 2008). The alpha/theta ratio has previously been used in several human EEG–fMRI studies (Horovitz et al., 2008; Laufs et al., 2006).
    #- The EEG alpha/theta time course was temporally aligned to the fMRI time course by removing the first seven timepoints. For analyses that directly correlate EEG with the fMRI alertness index (Figures 1, 2, 5, and 6), the aligned EEG alpha/theta time course was mean-centered and convolved with the default gamma-variate HRF provided in SPM (https://www.fil.ion.ucl.ac.uk/spm/) to account for approximate hemodynamic delays in relating EEG to fMRI, and band-pass filtered with nominal cutoffs from 0.01 to 0.2 Hz (using the bandpass function in MATLAB) to approximate the bandwidth of the fMRI data. The resulting signal is referred to as the ‘EEG alertness index’. For the analyses in Figure 4, Figure 4—figure supplement 2, we use the raw (not hemodynamically filtered) version of the EEG alpha/theta ratio.
    
    # sub-04_ses-01_task-rest_preprocess_eeg_v2.set
    #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_run-01_eeg_{eeg_preproc_step}.set")
    eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")
    if (os.path.exists(eeg_data) == False):
        eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")
    raw_eeg = mne.io.read_raw_edf(eeg_data, preload=True)
    current_sfreq = raw_eeg.info["sfreq"]
    print(f"freqency = {current_sfreq}")
    #print(raw_eeg.__dict__)

    #channel_names = (raw_eeg.ch_names)
    channel_names = ['P3', 'P4', 'Pz', 'O1', 'O2', 'Oz']
    #channel_names = ['Oz']

    events, event_id = mne.events_from_annotations(raw_eeg)
    #The first column contains the event time in samples. The third column contains the event id.
    scanner_tr_id = event_id['R128']
    print(f"events.shape {events.shape}, #R128(tr) {np.sum(events[:][:,2]==scanner_tr_id)}")
    sfreq = raw_eeg.info['sfreq']

    eeg_alpha = raw_eeg.filter(8, 12).copy()
    eeg_theta = raw_eeg.filter(3, 7).copy()

    event_id, tmin, tmax = scanner_tr_id, 0, (bold_tr-(1/sfreq)) # remove 1 sample, since bold_tr is new one
    #event_id, tmin, tmax = scanner_tr_id, -1.05, 1.05 # remove 1 sample, since bold_tr is new one
    alpha_epochs = mne.Epochs(eeg_alpha, events, event_id, tmin, tmax, 
        picks=channel_names, baseline=None, preload=True, decim=1,  
        event_repeated='merge')
    theta_epochs = mne.Epochs(eeg_theta, events, event_id, tmin, tmax, 
        picks=channel_names, baseline=None, preload=True, decim=1,  
        event_repeated='merge')
    

    alpha_epochs_data = np.nanmean(alpha_epochs.get_data(), axis=1)
    print(alpha_epochs_data.shape)
    theta_epochs_data = np.nanmean(theta_epochs.get_data(), axis=1)
    print(theta_epochs_data.shape)

    power_alpha_rms = np.sqrt(np.mean(alpha_epochs_data**2, axis=1))
    power_theta_rms = np.sqrt(np.mean(theta_epochs_data**2, axis=1))

    eeg_alertness_index = power_alpha_rms/power_theta_rms

    #bold_tr = 2.1
    fmri_time = np.arange(1.05,eeg_alertness_index.shape[0]*bold_tr,bold_tr)
    n_samples = np.round(eeg_alertness_index.shape[0] / ((1/bold_tr)/predHz)).astype('int') # tr 2.1sec to 1Hz - 1sec
    fmri_time_resampled = np.linspace(fmri_time[0], fmri_time[-1], num=n_samples)
    attidx_spline = CubicSpline(fmri_time, eeg_alertness_index)
    EEG_ALERTNESS_INDEX_RESAMPLED = attidx_spline(fmri_time_resampled)

    #plt.figure(); 
    #plt.plot( zscore(eeg_alertness_index) )

    return EEG_ALERTNESS_INDEX_RESAMPLED

