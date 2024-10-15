#%%
import pickle as pkl
import pandas as pd
import scipy.stats as sps
import numpy as np
import os
from glob import glob

task = "checker"
run = "run-01BlinksRemoved"
subject_list = [
    "sub-01_ses-01",
    "sub-02_ses-01"
    ]

ts_files_path = "/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models/data_prep/prediction_model_data_eeg_features_v2/group_data_Hz-1.0"

###############################################################################################

def xcorr_with_ttest(subject_list: list, 
                     sessions: list,
                     task: str,
                     run: str,
                     cap_name: str,
                     feature_name: str,
                     sampling_rate: float = 1.0):
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

    cap_list = ["tsCAP1","tsCAP2","tsCAP3","tsCAP4","tsCAP5","tsCAP6","tsCAP7","tsCAP8"]

    
    channel_list = ['Fp1','Fpz','Fp2','AF7','AF3','AF4','AF8','F7','F5','F3','F1','Fz',
                    'F2','F4','F6','F8','FT7','FT8','FC5','FC3','FC1','FC2','FC4','FC6',
                    'T7','T8','C5','C3','C1','Cz','C2','C4','C6','TP9','TP7','TP8','TP10',
                    'CP5','CP3','CP1','CPz','CP2','CP4','CP6','P7','P5','P3','P1','Pz',
                    'P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','O2','Oz']
    band_list = ["delta","theta","alpha","beta","gamma"]
    print(sessions)
    #initialize dictionary to hold t-tested xcorrs
    dict_xcorr_t = dict()

    #iterate over CAPs

    #iterate over channel-band combos
    for channel in channel_list:
        for band in band_list:
            #create label to be used for dict_xcorr_t keys
            channel_band = f"{channel}-{band}"

            #create dictionary to hold each subject's CAPxchannel-band xcorr data
            channel_band_xcorr_dict = dict()

            #iterate over subjects
            for subject in subject_list:
                #open subject's data
                ts_path = os.path.join(
                    ts_files_path, 
                    f"{subject}_task-{task}_run-{run}_multimodal_data.pkl"
                    )
                with open(ts_path, 'rb') as file:
                    data_dict = pkl.load(file)

                # get data for CAP
                cap_index = data_dict['brainstates']['labels'].index(cap_name)
                cap_data = data_dict['brainstates']['feature'][cap_index]

                #get mask for CAP
                cap_mask_float = data_dict['brainstates']['mask']
                cap_mask = cap_mask_float > 0.5                    

                #get data for channel-band
                ch_idx = data_dict[feature_name]['labels']["channels_info"]["channel_name"].index(channel)
                band_idx = band_list.index(band)

                eeg_ch_band_data = np.zeros(data_dict[feature_name]['feature'].shape[1])
                for tp in np.arange(eeg_ch_band_data.shape[0]):
                    eeg_ch_band_data[tp] = data_dict[feature_name]['feature'][ch_idx][tp][band_idx]
                
                #get mask for channel-band
                eeg_ch_band_mask = data_dict[feature_name]["mask"]
            
                #while BVA mystery artifact exists!
                if task == "checker": 
                    cap_data = cap_data[:682]
                    cap_mask = cap_mask[:682]
                    eeg_ch_band_data = eeg_ch_band_data[:682]
                    eeg_ch_band_mask = eeg_ch_band_mask[:682]
                if task == "rest":
                    cap_data = cap_data[:2204]
                    cap_mask = cap_mask[:2204]
                    eeg_ch_band_data = eeg_ch_band_data[:2204]
                    eeg_ch_band_mask = eeg_ch_band_mask[:2204]
                
                #calculate FMRI-EEG xcorr using CAP data, CAP mask, channel-band data, channel-band mask 
                window = int(30*sampling_rate)
                xcorr = np.zeros(window+1)
                for lag_idx, lag in enumerate(np.arange(int(window/2 * -1), int(window/2+1))):
                    cap_lag = np.zeros(cap_data.shape[0]-abs(lag))
                    cap_mask_lag = np.zeros(cap_mask.shape[0]-abs(lag))
                    if lag <= 0:
                        for idx in np.arange(cap_data.shape[0]-abs(lag)):
                            cap_lag[idx] = cap_data[idx-lag]
                            cap_mask_lag[idx] = cap_mask[idx-lag]
                        eeg_lag = eeg_ch_band_data[:cap_data.shape[0]+lag]   
                        eeg_mask_lag = eeg_ch_band_mask[:cap_data.shape[0]+lag]
                    if lag > 0:
                        for idx in np.arange(eeg_ch_band_data.shape[0]-abs(lag)):
                            cap_lag[idx] = cap_data[idx]
                            cap_mask_lag[idx] = cap_mask[idx]
                        eeg_lag = eeg_ch_band_data[lag:]
                        eeg_mask_lag = eeg_ch_band_mask[lag:]

                    joint_mask = cap_mask_lag==eeg_mask_lag
                    xcorr[lag_idx] = np.corrcoef(cap_lag[joint_mask], eeg_lag[joint_mask])[0,1]
                
                channel_band_xcorr_dict[subject] = xcorr
            
            #convert to dataframe for t-testing
            df_channel_band_xcorr = pd.DataFrame(channel_band_xcorr_dict)

            #t-test dataframe
            t_series = np.zeros(df_channel_band_xcorr.shape[0])
            for index, row in df_channel_band_xcorr.iterrows():
                t_stat, p_val = sps.ttest_1samp(row, popmean=0)
                t_series[index] = t_stat

            #add to master t-stat dictionary
            dict_xcorr_t[channel_band] = t_series


    #now CAPxPD!
    pd_analysis_list = ["pupil_dilation","first_derivative","second_derivative"]

    for pd_analysis in pd_analysis_list: 
        #create label to be used for dict_xcorr_t keys
        pd_analysis = f"{pd_analysis}"

        #create dictionary to hold each subject's CAPxPD-analysis xcorr data
        pd_xcorr_dict = dict()

        for subject in subject_list:
            ts_path = os.path.join(
                ts_files_path, 
                f"{subject}_task-{task}_run-{run}_multimodal_data.pkl"
                )
            with open(ts_path, 'rb') as file:
                data_dict = pkl.load(file)

            # get data for CAP
            cap_index = data_dict['brainstates']['labels'].index(cap_name)
            cap_data = data_dict['brainstates']['feature'][cap_index]

            #get mask for CAP
            cap_mask_float = data_dict['brainstates']['mask']
            cap_mask = cap_mask_float > 0.5

            #get PD data and calculate first, second deriv if necessary
            ftr_index = data_dict['pupil']['labels'].index("pupil_size")
            pd_data = data_dict['pupil']['feature'][ftr_index,:]
            if pd_analysis == "PD-firstder":
                pd_firstder_arr = np.zeros(pd_data.shape[0])
                pd_firstder_arr[:-1] = np.diff(pd_data)
                pd_data = pd_firstder_arr
            elif pd_analysis == "PD-secondder":
                pd_firstder_arr = np.zeros(pd_data.shape[0])
                pd_firstder_arr[:-1] = np.diff(pd_data)
                pd_secondder_arr = np.zeros(pd_data.shape[0])
                pd_secondder_arr[:-1] = np.diff(pd_firstder_arr)
                pd_data = pd_secondder_arr

            #get mask for PD
            pd_mask_float = data_dict['pupil']['mask']
            pd_mask = pd_mask_float > 0.5

            window = 120
            xcorr = np.zeros(window+1)
            for lag_idx, lag in enumerate(np.arange(int(window/2 * -1), int(window/2+1))):
                cap_lag = np.zeros(cap_data.shape[0]-abs(lag))
                cap_mask_lag = np.zeros(cap_mask.shape[0]-abs(lag))##
                if lag <= 0:
                    for idx in np.arange(cap_data.shape[0]-abs(lag)):
                        cap_lag[idx] = cap_data[idx-lag]
                        cap_mask_lag[idx] = cap_mask[idx-lag]##
                    pd_lag = pd_data[:cap_data.shape[0]+lag]#     
                    pd_mask_lag = pd_mask[:cap_data.shape[0]+lag]##
                if lag > 0:
                    for idx in np.arange(pd_data.shape[0]-abs(lag)):
                        cap_lag[idx] = cap_data[idx]
                        cap_mask_lag[idx] = cap_mask[idx]##
                    pd_lag = pd_data[lag:]#
                    pd_mask_lag = pd_mask[lag:]##

                joint_mask = cap_mask_lag==pd_mask_lag
                xcorr[lag_idx] = np.corrcoef(cap_lag[joint_mask], pd_lag[joint_mask])[0,1]#
                
            pd_xcorr_dict[subject] = xcorr
            
            #convert to dataframe for t-testing
            df_pd_xcorr = pd.DataFrame(pd_xcorr_dict)

            #t-test dataframe
            t_series = np.zeros(df_pd_xcorr.shape[0])
            for index, row in df_pd_xcorr.iterrows():
                t_stat, p_val = sps.ttest_1samp(row, popmean=0)
                t_series[index] = t_stat

            #add to master t-stat dictionary
            dict_xcorr_t[pd_analysis] = t_series
    
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
                       
                       
###############################################################################################

#%%
dict_xcorr_t = xcorr_with_ttest(subject_list, 
                                task = "checker",
                                sessions = None,
                                cap_name = "tsCAP1",
                                sampling_rate=1.0,
                                feature_name = "EEGbandsEnvelopes",
                                run = "01")