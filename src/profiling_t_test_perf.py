
import os

nthreads = "32" # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
import scipy.stats
import numpy as np
import pandas as pd
from pyinstrument import Profiler

def xcorr_with_ttest(subject_list: list, 
                     eeg_files_path: str | os.PathLike,
                     pupil_files_path: str | os.PathLike,
                     sessions: list,
                     task: str,
                     run: str,
                     cap_name: str,
                     sampling_rate: str,
                     nb_features = 5,
                     ):
    """ Returns dictionary of t-tested cross correlations. 
    Returns dictionary of t-tested cross correlations for a given list of 
    participants' fMRI, EEG, and pupillometry data from a given task.  
    Each element in the dictionary is a series of t-stats for a certain CAP x 
    channel-band or CAP x pupillometry analysis cross correlation 
    (and is labeled as such, eg "tsCAP1_Oz-alpha", "tsCAP5_PD-firstder")
    
    Args:
        subject_list (list): The list of subject to run the function on.
        sessions (list): The list of sessions per subject to run the function on.
        task (str): The 
        run (str): _description_

    Returns:
        _type_: _description_
    """
    
    channel_list = ['Fp1','Fpz','Fp2','AF7','AF3','AF4','AF8','F7','F5','F3','F1','Fz',
                    'F2','F4','F6','F8','FT7','FT8','FC5','FC3','FC1','FC2','FC4','FC6',
                    'T7','T8','C5','C3','C1','Cz','C2','C4','C6','TP9','TP7','TP8','TP10',
                    'CP5','CP3','CP1','CPz','CP2','CP4','CP6','P7','P5','P3','P1','Pz',
                    'P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','O2','Oz']
    band_list = ["delta","theta","alpha","beta","gamma"]

    #initialize dictionary to hold t-tested xcorrs

    pupil_analysis_list = ["PD","PD-firstder","PD-secondder"]
    pupil_analysis_keys = ["pupil_dilation","first_derivative","second_derivative"]
    chan_populating_list = list()
    #iterate over channel-band combos
    for channel in channel_list:
        band_populating_list = list()
        for band in band_list:
            #create label to be used for dict_xcorr_t keys
            channel_band = f"{channel}-{band}"

            #open group data
            ts_path = os.path.join(
                eeg_files_path,
                sampling_rate, 
                f"task-{task}_{cap_name}_{channel}-{band}-raw_xcorr-full.csv")
            try:
                df_ts = pd.read_csv(ts_path)
                sub_idx_in_df = [col for col in df_ts.columns
                                 if col.split('_')[0] in subject_list]
                selection = df_ts[sub_idx_in_df]
            except Exception as e:
                raise e
                continue
            band_populating_list.append(selection.to_numpy().T)
        
        bands_array = np.stack(band_populating_list,axis = 2)
        chan_populating_list.append(bands_array)
    chan_array = np.stack(chan_populating_list,axis = 1)

    pupil_populating_list = list()
    for pupil_analysis in pupil_analysis_list: 
        #open group data
        ts_path = os.path.join(
            pupil_files_path,
            pupil_analysis, 
            f"task-{task}_{cap_name}_{pupil_analysis}_xcorr-full.csv")
        try:
            df_ts = pd.read_csv(ts_path)
            selection = df_ts[sub_idx_in_df]
            #open xcorr dataframe
        except Exception as e:
            #raise e
            continue   

        pupil_populating_list.append(selection.to_numpy().T)
    pupil_array = np.stack(pupil_populating_list, axis = 1)
    pupil_array = np.expand_dims(pupil_array, axis = 3)
    pupil_array = np.repeat(pupil_array, 5, axis = -1)
    assembled_array = np.concatenate([chan_array, pupil_array], axis = 1)
    t_stat, _ = scipy.stats.ttest_1samp(assembled_array, popmean=0, axis = 0)

    channel_list = channel_list + pupil_analysis_keys
    max_array = np.max(t_stat,axis = 1)
    sorted_array = np.argsort(max_array, axis = None)[::-1]
    index_matrix = np.stack(np.unravel_index(sorted_array,max_array.shape), axis = 0)
    #for channel, band in 
    dict_xcorr_t = dict()

    for pupil_index in [61,62,63]:
        idx = np.where(index_matrix[0,:] == pupil_index)[0][1:]
        index_matrix = np.delete(index_matrix,idx, axis = 1)
    
    if any(index_matrix[0,:nb_features] >= 61):
        dict_xcorr_t['pupil'] = list()
    if any(index_matrix[0,:nb_features] < 61):
        dict_xcorr_t['EEGbandsEnvelopes'] = {
            'channel' : list(),
            'band' : list()
        } 
    for feat_idx in range(nb_features):
        chan, band = (channel_list[index_matrix[0,feat_idx]] ,
                      band_list[index_matrix[1, feat_idx]])
        
        if index_matrix[0,feat_idx] >= 61:
            dict_xcorr_t['pupil'].append(chan)
        else:
            dict_xcorr_t['EEGbandsEnvelopes']['channel'].append(chan)
            dict_xcorr_t['EEGbandsEnvelopes']['band'].append(band)
        
    return dict_xcorr_t       

study_directory = (
    "/data2/Projects/eeg_fmri_natview/derivatives"
    "/multimodal_prediction_models/data_prep"
    f"/prediction_model_data_eeg_features_checker/group_data_Hz-3.8"
    )

rand_generator = np.random.default_rng()
caps = np.array(['tsCAP1',
        'tsCAP2',
        'tsCAP3',
        'tsCAP4',
        'tsCAP5',
        'tsCAP6',
        'tsCAP7',
        'tsCAP8'])

runs = ['01']#, '02']
tasks = ['checker']#['tp','dme','monkey1','monkey2','monkey5','inscape']
SAMPLING_RATE_HZ = 3.8
WINDOW_LENGTH_SECONDS = 10
sessions = ["01","02"]
subjects = ["sub-01","sub-02",
            "sub-03","sub-04",
            "sub-05","sub-06",
            "sub-07","sub-08",
            "sub-09","sub-10",]


subject_list=subjects
sessions = ['01','02']
task = 'checker'
run = '01'
cap_name='tsCAP1'
sampling_rate='3.8-Hz'
nb_features = 3
with Profiler() as profiler:
    dict_xcorr_t = xcorr_with_ttest(
        subject_list=subject_list,
        eeg_files_path=f'/home/thoppe/fmri-analysis/natview/FMRI-EEG-files/xcorr-data/{task}/cap-band-full-truncated/raw',
        pupil_files_path = f'/home/thoppe/fmri-analysis/natview/FMRI-PD-files/xcorr-data/{task}',
        sessions = sessions,
        task = task,
        run = run,
        cap_name = cap_name,
        sampling_rate=str(sampling_rate))

profiler.write_html('output_performance.html', 
                    timeline=True, show_all=True)