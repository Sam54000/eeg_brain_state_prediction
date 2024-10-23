def xcorr_with_ttest(subject_list: list, 
                     ts_files_path: str | os.PathLike,
                     sessions: list,
                     task: str,
                     run: str,
                     cap_name: str,
                     sampling_rate: float,
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
                ts_files_path,
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
                        ts_dict_abrgd[f"sub-{subject}_ses-{session}"] = df_ts[f"sub-{subject}_ses-{session}_run-{run}"]
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
    pd_analysis_list = ["pupil_dilation","first_derivative","second_derivative"]

    for pd_analysis in pd_analysis_list: 
        #create label to be used for dict_xcorr_t keys
        pd_analysis = f"{pd_analysis}"

        pd_xcorr_dict = dict()

        #open group data
        ts_path = os.path.join(
            ts_files_path,
            sampling_rate, 
            f"task-{task}_{cap_name}_{channel}-{band}-raw_{sampling_rate}_xcorr.csv")
        try:
            True
            #open xcorr dataframe
        except Exception as e:
            #raise e
            continue   

        #iterate over subjects to create abridged dataframe
        ts_dict_abrgd = dict()

        for subject in subject_list:
            for session in sessions:
                try:
                    ts_dict_abrgd[f"sub-{subject}_ses-{session}_run-{run}"] = df_ts[f"sub-{subject}_ses-{session}"]
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
        dict_xcorr_t[pd_analysis] = t_series
    
    return dict_xcorr_t