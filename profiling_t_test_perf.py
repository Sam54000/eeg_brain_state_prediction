
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
    dict_xcorr_t = dict()

    #iterate over channel-band combos
    for channel in channel_list:
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
            except Exception as e:
                #raise e
                continue

            #iterate over subjects to create abridged dataframe
            ts_dict_abrgd = dict()

            for subject in subject_list:
                for session in sessions:
                    try:
                        ts_dict_abrgd[f"{subject}_ses-{session}_run-{run}"] = df_ts[f"{subject}_ses-{session}"]
                    except Exception as e:
                        #raise e
                        continue
            
            df_ts_abrgd = pd.DataFrame(ts_dict_abrgd)

            #t-test abridged dataframe
            t_series = np.zeros(df_ts_abrgd.shape[0])
            for index, row in df_ts_abrgd.iterrows():
                t_stat, p_val = scipy.stats.ttest_1samp(row, popmean=0)
                t_series[index] = t_stat

            #add to master t-stat dictionary
            dict_xcorr_t[channel_band] = t_series


    #now CAPxPD!
    pupil_analysis_list = ["PD","PD-firstder","PD-secondder"]
    pupil_analysis_keys = ["pupil_dilation","first_derivative","second_derivative"]

    for pupil_analysis, pupil_key in zip(pupil_analysis_list, pupil_analysis_keys): 
        #open group data
        ts_path = os.path.join(
            pupil_files_path,
            pupil_analysis, 
            f"task-{task}_{cap_name}_{pupil_analysis}_xcorr-full.csv")
        try:
            df_ts = pd.read_csv(ts_path)
            #open xcorr dataframe
        except Exception as e:
            #raise e
            continue   

        #iterate over subjects to create abridged dataframe
        ts_dict_abrgd = dict()

        for subject in subject_list:
            for session in sessions:
                try:
                    ts_dict_abrgd[f"{subject}_ses-{session}_run-{run}"] = df_ts[f"{subject}_ses-{session}"]
                except Exception as e:
                    #raise e
                    continue
        
        df_ts_abrgd = pd.DataFrame(ts_dict_abrgd)

        #t-test abridged dataframe
        t_series = np.zeros(df_ts_abrgd.shape[0])
        for index, row in df_ts_abrgd.iterrows():
            t_stat, p_val = scipy.stats.ttest_1samp(row, popmean=0)
            t_series[index] = t_stat

        #add to master t-stat dictionary
        dict_xcorr_t[pupil_key] = t_series
    
    return dict_xcorr_t

def select_feature(dict_xcorr_t: dict,
                   nb_features: int = 5) -> dict:
    abs_max_t = dict()
    for key, values in dict_xcorr_t.items():
        abs_max_t[key] = np.max(np.abs(values))
    
    sorted_dict = dict(sorted(abs_max_t.items(), 
                              key=lambda item: item[1], 
                              reverse=True)
    )
    
    return list(sorted_dict.keys())[:nb_features]

def format_features_info(features_info: list):
    output_info = {"EEGbandsEnvelopes":{
        "channel":list(),
        "band":list()
    },
                   "pupil":list()
    }
    
    for feature_info in features_info:
        if "pupil" in feature_info or "derivative" in feature_info:
           output_info["pupil"].append(feature_info)
           
        else:
            channel, band = feature_info.split('-')
            output_info["EEGbandsEnvelopes"]["channel"].append(channel)
            output_info["EEGbandsEnvelopes"]["band"].append(band)
    
    
    
    return {key: value for key, value in output_info.items() if value}

def main_xcorr(subject_list:list,
                sessions: list,
                task: str,
                run: str,
                cap_name: str,
                sampling_rate: str = '3.8-Hz',
                nb_features: int = 5) -> dict:
    
    dict_xcorr_t = xcorr_with_ttest(
        subject_list=subject_list,
        eeg_files_path=f'/home/thoppe/fmri-analysis/natview/FMRI-EEG-files/xcorr-data/{task}/cap-band-full-truncated/raw',
        pupil_files_path = f'/home/thoppe/fmri-analysis/natview/FMRI-PD-files/xcorr-data/{task}',
        sessions = sessions,
        task = task,
        run = run,
        cap_name = cap_name,
        sampling_rate=str(sampling_rate))
    
    selected_features = select_feature(dict_xcorr_t=dict_xcorr_t,
                                       nb_features=nb_features)
    formated_features_info = format_features_info(selected_features)
    
    return formated_features_info  

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