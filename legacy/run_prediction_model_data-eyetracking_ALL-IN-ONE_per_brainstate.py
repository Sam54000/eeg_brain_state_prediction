#%% =======================================================================================
#%load_ext autoreload
#%autoreload 2
#%env OMP_NUM_THREADS=16

import os

nthreads = "64" # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from scipy.stats import zscore
import seaborn as sns
from nilearn.signal import butterworth

#import prediction_functions as pf
import importlib
#importlib.reload(pf)

#%% =======================================================================================

base_dir = '/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models_with_legacy_code'

predHz = 3.8 #0.5# 3.8
do_HRF = 0
n_feat = 38 #24 # 38
print(f"-{n_feat/predHz}sec")

base_data_dir = os.path.join(base_dir,'data_prep', 'prediction_model_data_eyetracking_all_tasks_fixed')
data_dir = os.path.join(base_data_dir, f"group_data_Hz-{predHz}")

base_out_dir = os.path.join(base_dir, "analysis_all_movies")
out_dir = os.path.join(base_out_dir, f"prediction_results_eyetracking_HRF-{do_HRF}_Hz-{predHz}_nfeats-{n_feat}")
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

#%% =======================================================================================

# load data
print("loading...") # added last 2
file_str = f"group_data_natview_data_fmri_eyetracking"
filename = os.path.join(data_dir,f"{file_str}.npy")
data = np.load(filename, allow_pickle=True)
data = data.item()
print("...done")

# print( data.keys() )
# -------------------------

subjects = data['subjects']
sessions = data['sessions']
tasks = data['tasks']

data_brainstates = data['data_fmri']
data_eyetracking = data['data_eyetracking']
# data_respiration = data['data_respiration']
# data_eeg_bands_envelope = data['data_eeg_bands_envelope']
# data_eeg_custom_envelope = data['data_eeg_custom_envelope']
# data_eeg_morlet_tfr = data['data_eeg_morlet_tfr']

# tmp_key = 'sub-01_ses-01_task-checker_run-01'
# #(EEGbandsEnvelopes_resampled, fmri_time_resampled, EEGbandsEnvelopes_data['frequencies'], EEGbandsEnvelopes_data['channels_info'])
# the_eeg_features = data_eeg_bands_envelope
# electrode_labels = the_eeg_features[tmp_key][3]
# iter_freqs = the_eeg_features[tmp_key][2]
# iter_freqs_labels = ["0.5-4Hz", "4-8Hz", "8-13Hz", "13-30Hz", "30-40Hz"]

# #electrode_labels['index']
# #np.where(np.array(electrode_labels['channel_name']) == 'Oz')[0][0]
# #electrode_labels['channel_name'][19]


#%% =======================================================================================

subjects
sessions
#tasks = ['checker_run-01', 'rest_run-01'] #  ['checker_run-01', 'rest_run-01', 'tp_run-01', 'tp_run-02'] # 
#tasks = ['checker_run-01', 'rest_run-01', 'tp_run-01', 'tp_run-02'] # 
#tasks = ['checker_run-01', 'rest_run-01'] # , 'tp_run-01', 'tp_run-02'] # 
#tasks = ['checker_run-01'] # , 'tp_run-01', 'tp_run-02'] # 

# 'checker_run-01', 'rest_run-01', 
#tasks = [ 'inscapes_run-01', 
#            'dme_run-01', 'dme_run-02', 'monkey1_run-01', 'monkey1_run-02', 'tp_run-01', 'tp_run-02', 
#            'dmh_run-01', 'dmh_run-02', 'monkey2_run-01', 'monkey2_run-02', 'monkey5_run-01', 'monkey5_run-02' ]
tasks = ['checker_run-01']

# ['time',  'tsMask', 
# 'tsCAP1', 'tsCAP2', 'tsCAP3', 'tsCAP4', 'tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8',
# 'yeo7net1', 'yeo7net2', 'yeo7net3', 'yeo7net4', 'yeo7net5', 'yeo7net6', 'yeo7net7', 
# 'yeo17net1', 'yeo17net2', 'yeo17net3', 'yeo17net4', 'yeo17net5', 'yeo17net6', 'yeo17net7', 
# 'yeo17net8', 'yeo17net9', 'yeo17net10', 'yeo17net11', 'yeo17net12', 'yeo17net13', 'yeo17net14',
# 'yeo17net15', 'yeo17net16', 'yeo17net17', 
# 'GS']

#'yeo17net1', 'yeo17net2', 'yeo17net3', 'yeo17net4', 'yeo17net5', 'yeo17net6', 'yeo17net7', 
#'yeo17net8', 'yeo17net9', 'yeo17net10', 'yeo17net11', 'yeo17net12', 'yeo17net13', 'yeo17net14',
#'yeo17net15', 'yeo17net16', 'yeo17net17'

#brainstates = [('cap_ts',8)]
brainstates = ['tsCAP1']#, 'tsCAP2', 'tsCAP3', 'tsCAP4', 'tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8']
#brainstates = [ 'GS', 'GS_raw',
#'tsCAP1', 'tsCAP2', 'tsCAP3', 'tsCAP4', 'tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8',
#'yeo7net1', 'yeo7net2', 'yeo7net3', 'yeo7net4', 'yeo7net5', 'yeo7net6', 'yeo7net7' ]


test_sub = "01"
test_ses = "01"
test_task = 'checker_run-01' # rest_run tp_run checker_run tp_run-02 
test_bstate = 'tsCAP1' # ['peak_voxel', 'cap_ts', 'pca_cap_ts', 'extracted_ts']
#myBStateNum = 5
test_key = f"sub-{test_sub}_ses-{test_ses}_task-{test_task}"
data_brainstates.keys()


infos_dict = dict()
infos_dict['subjects'] = subjects
infos_dict['sessions'] = sessions
infos_dict['tasks'] = tasks
infos_dict['brainstates'] = brainstates

RESULTS = dict()
RESULTS['infos'] = infos_dict

pred_results = dict()


total_count = 0
for test_bstate in brainstates:
    pred_results_dir = os.path.join(out_dir, f"all_movies_bstate-{test_bstate}_nfeats-{n_feat}")
    if not os.path.exists(pred_results_dir):
        os.makedirs(pred_results_dir)
    # --------------------------------------------------------------------------------------------------
    # STACK ALL THE DATA FIRST AND GRAB IT LATER ACCORDINGLY FOR TRAIN/TEST

    SUBJECT_KEYS = np.array([])
    STACKED_FEATURES = np.array([])
    STACKED_TARGETS = np.array([])
    STACKED_FEATURES_MASK = np.array([])
    STACKED_TARGETS_MASK = np.array([])

    tmp_SUBJECT_KEYS = np.array([])
    tmp_STACKED_FEATURES = np.array([])
    tmp_STACKED_TARGETS = np.array([])
    tmp_STACKED_FEATURES_MASK = np.array([])
    tmp_STACKED_TARGETS_MASK = np.array([])

    for test_task in tasks:
        print(f"===============================================")    
        print(f"- {test_task}")

        # --- STACK HERE -------------------------------------------------------------------------
        key_idx = 0
        key = 'sub-01_ses-01_task-checker_run-01'
        for key_idx, key in enumerate(data_eyetracking.keys()):
            #print(f"===============================================")    
            #print(f"- {key}")
            total_count = total_count + 1

            if (key.split('_')[2].split('-')[1] != test_task.split('_')[0]):
                #print(f"- wrong task...")
                continue
            
            try:
                #data_eyetracking[test_key]
                #data_brainstates[test_key][test_bstate]
                data_eyetracking[key]
                data_brainstates[key][test_bstate]
            except:
                #print(f"- some data is missing")
                continue

            
            do_concat = 1

            fmri_ts_mask = (data_brainstates[key]['tsMask'].to_numpy() > 0.5)
            eyetrack_mask = (data_eyetracking[key]['tmask'].to_numpy() > 0.5)

            FMRI_TARGET_DATA = data_brainstates[key][test_bstate].to_numpy()
            PD_DATA = data_eyetracking[key]['pupil_size'].to_numpy()
            Xpos = data_eyetracking[key]['X_position'].to_numpy()
            Ypos = data_eyetracking[key]['Y_position'].to_numpy()
            
            tr_times = np.arange(0, 30, 1/predHz)
            #hrf_at_trs = pf.hrf(tr_times)
            if do_HRF:
                PD_DATA = np.convolve(PD_DATA, hrf_at_trs)
                #for ii in np.arange(EEG_FEATURES.shape[1]):
                #    tmp = np.convolve(EEG_FEATURES[:, ii], hrf_at_trs)
                #    EEG_FEATURES[:, ii] = tmp[:EEG_FEATURES.shape[0]]

            #PD_DATA = butterworth(PD_DATA, predHz, low_pass=0.1, high_pass=0.01, copy=True)
            PD_DATA_dfirst = np.diff(PD_DATA, prepend=PD_DATA[0])
            PD_DATA_dsecond = np.diff(PD_DATA, n=2, prepend=PD_DATA_dfirst[:2])
            #PD_DATA_dsecond[0] = 0
            #PD_DATA_dsecond[1] = 0
            PREDICTION_FEATURES = np.vstack((PD_DATA, PD_DATA_dfirst, PD_DATA_dsecond))

            # remove the first 5 seconds
            start_index = int(predHz*5)
            FMRI_TARGET_DATA = FMRI_TARGET_DATA[start_index:]
            PREDICTION_FEATURES = PREDICTION_FEATURES[:, start_index:]
            fmri_ts_mask = fmri_ts_mask[start_index:]
            eyetrack_mask = eyetrack_mask[start_index:]

            print(f"Prediction features - {PREDICTION_FEATURES.shape}")
            PREDICTION_FEATURES = zscore(PREDICTION_FEATURES.T).T
            print(f"Prediction features reshaped - {PREDICTION_FEATURES.shape}")

            #print(f"n_fmri : {FMRI_TARGET_DATA.shape}; n_eeg : {PREDICTION_FEATURES.shape}")
            npds, nn = PREDICTION_FEATURES.shape

            
            THE_ONE_MASK = fmri_ts_mask
            FEATURE_MASK = eyetrack_mask
            #FEATURE_MASK = np.ones(fmri_ts_mask.shape)

            FMRI_TARGETS = FMRI_TARGET_DATA[n_feat:]
            FMRI_TARGETS_MASK = THE_ONE_MASK[n_feat:]
            n_ts = FMRI_TARGET_DATA.shape[0]
              # stack everything
            #tmp_keys = np.repeat(key, FMRI_TARGETS.shape)
            tmp_keys = np.repeat(f"{key}", FMRI_TARGETS.shape)
            tmp_SUBJECT_KEYS = np.hstack([tmp_SUBJECT_KEYS, tmp_keys]) if tmp_SUBJECT_KEYS.size else tmp_keys
            tmp_STACKED_TARGETS = np.hstack([tmp_STACKED_TARGETS, FMRI_TARGETS]) if tmp_STACKED_TARGETS.size else FMRI_TARGETS
            tmp_STACKED_TARGETS_MASK = np.hstack([tmp_STACKED_TARGETS_MASK, FMRI_TARGETS_MASK]) if tmp_STACKED_TARGETS_MASK.size else FMRI_TARGETS_MASK
            for ii in np.arange(n_feat,n_ts,1):
                X = np.reshape(PREDICTION_FEATURES[:, (ii-n_feat):(ii)].flatten(),(1,-1))
                tmp_STACKED_FEATURES = np.vstack([tmp_STACKED_FEATURES, X]) if tmp_STACKED_FEATURES.size else X     
                tmp_feature_mask = FEATURE_MASK[(ii-n_feat):(ii)]
                tmp_STACKED_FEATURES_MASK = np.vstack([tmp_STACKED_FEATURES_MASK, tmp_feature_mask]) if tmp_STACKED_FEATURES_MASK.size else tmp_feature_mask     

            # end data will not be in there! #  | (key_idx == (len(data_eyetracking.keys())-1))
            if ((key_idx % 10) == 0):
                SUBJECT_KEYS = np.concatenate((SUBJECT_KEYS, tmp_SUBJECT_KEYS)) if SUBJECT_KEYS.size else tmp_SUBJECT_KEYS
                STACKED_FEATURES = np.concatenate((STACKED_FEATURES, tmp_STACKED_FEATURES)) if STACKED_FEATURES.size else tmp_STACKED_FEATURES
                STACKED_TARGETS = np.concatenate((STACKED_TARGETS, tmp_STACKED_TARGETS)) if STACKED_TARGETS.size else tmp_STACKED_TARGETS
                STACKED_FEATURES_MASK = np.concatenate((STACKED_FEATURES_MASK, tmp_STACKED_FEATURES_MASK)) if STACKED_FEATURES_MASK.size else tmp_STACKED_FEATURES_MASK
                STACKED_TARGETS_MASK = np.concatenate((STACKED_TARGETS_MASK, tmp_STACKED_TARGETS_MASK)) if STACKED_TARGETS_MASK.size else tmp_STACKED_TARGETS_MASK
                tmp_SUBJECT_KEYS = np.array([])
                tmp_STACKED_FEATURES = np.array([])
                tmp_STACKED_TARGETS = np.array([])
                tmp_STACKED_FEATURES_MASK = np.array([])
                tmp_STACKED_TARGETS_MASK = np.array([])
                #print("- concatenated")
                do_concat = 0

        # add last stuff that was maybe omitted
        if do_concat:
                SUBJECT_KEYS = np.concatenate((SUBJECT_KEYS, tmp_SUBJECT_KEYS)) if SUBJECT_KEYS.size else tmp_SUBJECT_KEYS
                STACKED_FEATURES = np.concatenate((STACKED_FEATURES, tmp_STACKED_FEATURES)) if STACKED_FEATURES.size else tmp_STACKED_FEATURES
                STACKED_TARGETS = np.concatenate((STACKED_TARGETS, tmp_STACKED_TARGETS)) if STACKED_TARGETS.size else tmp_STACKED_TARGETS
                STACKED_FEATURES_MASK = np.concatenate((STACKED_FEATURES_MASK, tmp_STACKED_FEATURES_MASK)) if STACKED_FEATURES_MASK.size else tmp_STACKED_FEATURES_MASK
                STACKED_TARGETS_MASK = np.concatenate((STACKED_TARGETS_MASK, tmp_STACKED_TARGETS_MASK)) if STACKED_TARGETS_MASK.size else tmp_STACKED_TARGETS_MASK
                tmp_SUBJECT_KEYS = np.array([])
                tmp_STACKED_FEATURES = np.array([])
                tmp_STACKED_TARGETS = np.array([])
                tmp_STACKED_FEATURES_MASK = np.array([])
                tmp_STACKED_TARGETS_MASK = np.array([])
                #print("- concatenated")
                do_concat = 0

    print(f"STACKED_FEATURES - {STACKED_FEATURES.shape}")
    print(total_count)
    # --- STACK HERE -------------------------------------------------------------------------
    # --- MASK IT BASED ON MISSING DATA IN FEATURES ------------------------------------------
    n_features_have_data = np.sum(STACKED_FEATURES_MASK, axis=1)
    n_feat_thrsh = (n_feat)*0.75
    STACKED_TARGETS_MASK[ n_features_have_data < n_feat_thrsh ] = False

    STACKED_TARGETS_MASK_ORIG = STACKED_TARGETS_MASK.copy()
    SUBJECT_KEYS_ORIG = SUBJECT_KEYS.copy()

    SUBJECT_KEYS = SUBJECT_KEYS[STACKED_TARGETS_MASK]
    STACKED_FEATURES = STACKED_FEATURES[STACKED_TARGETS_MASK, :]
    STACKED_TARGETS = STACKED_TARGETS[STACKED_TARGETS_MASK]

    print(f"STACKED_TARGETS_MASK - {STACKED_TARGETS_MASK.shape}")
    print(f"SUBJECT_KEYS - {SUBJECT_KEYS.T.shape}")
    print(f"STACKED_FEATURES - {STACKED_FEATURES.shape}")
    print(f"STACKED_TARGETS - {STACKED_TARGETS.T.shape}")
    # --- MASK IT BASED ON MISSING DATA ETC -----------------------------------------------------
    # --- STACKING AND MASKING ENDS HERE --------------------------------------------------------
        
    test_sub = "22"
    test_ses = "02"
    test_task = 'monkey5_run-01'
    #tasks = [ 'inscapes_run-01', 
    #    'dme_run-01', 'dme_run-02', 'monkey1_run-01', 'monkey1_run-02', 'tp_run-01', 'tp_run-02', 
    #    'dmh_run-01', 'dmh_run-02', 'monkey2_run-01', 'monkey2_run-02', 'monkey5_run-01', 'monkey5_run-02' ]
    for test_task in tasks:
        predictions_run = dict()
        # --- NOW SLPIT IT IN TRAIN AND TEST AND RUN THE PREDICION ----------------------------------
        for test_sub in subjects:
            for test_ses in sessions:
                plt.close('all')
                print(f"- {test_sub} - {test_ses} - {test_task}")
                print(f"===============================================")
                #test_key = f"sub-{test_sub}_ses-{test_ses}_task-{test_task}"
                test_key = f"sub-{test_sub}_ses-{test_ses}_task-{test_task}"

                test_subject_index = (SUBJECT_KEYS == test_key)
                #SUBJECT_KEYS[test_subject_index]
                train_subject_index = np.frompyfunc(lambda x: f"sub-{test_sub}" in x, 1, 1)(SUBJECT_KEYS)

                TRAINING_EEG_FEATURES = STACKED_FEATURES[ train_subject_index==False, :]
                TEST_EEG_FEATURES = STACKED_FEATURES[ test_subject_index==True, :]

                TRAINING_FMRI_TARGETS = STACKED_TARGETS[ train_subject_index==False ]
                TEST_FMRI_TARGETS = STACKED_TARGETS[ test_subject_index==True ]

                test_subject_orig_index = (SUBJECT_KEYS_ORIG == test_key)
                train_subject_orig_index = np.frompyfunc(lambda x: f"sub-{test_sub}" in x, 1, 1)(SUBJECT_KEYS_ORIG)
                MASK_TRAINING_DATA = STACKED_TARGETS_MASK_ORIG[ train_subject_orig_index==False ]
                MASK_TEST_DATA = STACKED_TARGETS_MASK_ORIG[ test_subject_orig_index ]

                print(f"TRAINING_EEG_FEATURES - {TRAINING_EEG_FEATURES.shape}")
                print(f"TRAINING_FMRI_TARGETS - {TRAINING_FMRI_TARGETS.T.shape}")
                print(f"TEST_EEG_FEATURES - {TEST_EEG_FEATURES.shape}")
                print(f"TEST_FMRI_TARGETS - {TEST_FMRI_TARGETS.T.shape}")
                print(f"SUM MASK_TEST_DATA - {np.sum(MASK_TEST_DATA)}")
                print(f"MASK_TEST_DATA_ORIG - {MASK_TEST_DATA.shape}")
                #print(f"MASK_TRAINING_DATA - {MASK_TRAINING_DATA.T.shape}")
                 
                # skip if no test data
                if (TEST_FMRI_TARGETS.shape[0] < 10):
                    continue

                # --- PREDICTION MODEL HERE ------------------------------------------------------
                
                feature_scale = 1 # 1e10

                # TRAINING -----------------------------------------------------------------------------------
                # fit the model
                Xtrain = TRAINING_EEG_FEATURES * feature_scale
                ytrain = TRAINING_FMRI_TARGETS.T

                #reg = linear_model.LassoCV()
                #reg = linear_model.Ridge(alpha=1)
                reg = linear_model.RidgeCV(cv=5)
                #reg = linear_model.ElasticNetCV(random_state=0, l1_ratio=0.99)
                reg.fit(Xtrain, ytrain)

                coefs = reg.coef_

                #f = plt.figure(figsize=(10, 3))
                #plt.plot(zscore(ytrain), zorder=1, linewidth=2, color='k', label='test data') 
                #plt.plot(zscore(reg.predict(Xtrain)), zorder=3, linewidth=2, linestyle='-', label='prediction', color='tomato')  
                
                # TESTING -----------------------------------------------------------------------------------

                Xtest = TEST_EEG_FEATURES * feature_scale
                ytest = TEST_FMRI_TARGETS.T
                yhat = reg.predict(Xtest)

                foo = np.corrcoef(ytest.T,yhat.T)
                prediction_elec_targets = foo[0,1]
                print(prediction_elec_targets)

                ytest_full = np.zeros(MASK_TEST_DATA.shape) * np.nan
                ytest_full[MASK_TEST_DATA] = zscore(ytest)
                yhat_full = np.zeros(MASK_TEST_DATA.shape) * np.nan
                yhat_full[MASK_TEST_DATA] = zscore(yhat)

                f = plt.figure(figsize=(10, 3))
                ax1 = plt.subplot(1,3,(1,2))
                plt.plot(ytest_full, zorder=1, linewidth=2, color='k', label='test data') 
                plt.plot(yhat_full, zorder=3, linewidth=2, linestyle='-', label='prediction', color='tomato')  
                #plt.plot(zscore(ytest), zorder=1, linewidth=2, color='k', label='test data') 
                #plt.plot(zscore(yhat), zorder=3, linewidth=2, linestyle='-', label='prediction')  
                plt.legend(frameon=False)                
                plt.suptitle(f"{test_bstate} - testset: {test_key} : r={foo[0,1]:.2f}", y=0.9, va='center')
                ax1 = plt.subplot(1,3,3)
                coefs_reshaped = np.reshape(coefs,(npds, (n_feat)))
                vmax = np.max(np.abs(coefs_reshaped))
                h = plt.imshow(coefs_reshaped, aspect='auto', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
                plt.colorbar(h)
                yticks = np.arange(npds)
                yticks_labels = np.tile(["PD","dPD","ddPD"],1)
                plt.yticks(ticks=yticks,labels=yticks_labels, rotation=0)
                tsteps = 16 if n_feat>10 else 1
                #xtickpos = [0, coefs_reshaped.shape[1]/3, (coefs_reshaped.shape[1]/3)*2, (coefs_reshaped.shape[1]-1)]
                #xticklabels = [ f"-{n_feat/predHz}sec", f"-{((coefs_reshaped.shape[1]/3)*2)/predHz}sec", f"-{((coefs_reshaped.shape[1]/3))/predHz}sec", "0"]
                xtickpos = [0, coefs_reshaped.shape[1]/2, (coefs_reshaped.shape[1]-1)]
                xticklabels = [ f"-{n_feat/predHz}sec", f"-{((coefs_reshaped.shape[1]/2))/predHz}sec", "0"]
                plt.xticks(ticks=xtickpos,labels=xticklabels)
                f.tight_layout()

                plot_str = f"{test_bstate}_prediction_testset_trfeats-{n_feat}_{test_key}"
                plt.savefig(os.path.join(pred_results_dir,f"{plot_str}.png"), dpi=300, bbox_inches="tight")
                #plt.savefig(os.path.join(pred_results_dir,f"{plot_str}.svg"), dpi=300, bbox_inches="tight") 

                if (test_key not in predictions_run):
                    predictions_run[test_key] = {}
                predictions_run[test_key] = (ytest, yhat, prediction_elec_targets, coefs_reshaped, MASK_TEST_DATA)

                #['ytest'] ['yhat'] ['pred_r'] ['coefs_reshaped']
                if (test_key not in pred_results):
                    pred_results[test_key] = {}
                pred_results[test_key][test_bstate] = (ytest, yhat, prediction_elec_targets, coefs_reshaped, MASK_TEST_DATA)

                # --- PREDICTION MODEL HERE ------------------------------------------------------
                
                
        # # --- END OF THE PREDICION ------------------------------------------------------------------           
        # print("saving...")
        # file_str = f"prediction_results_{test_task}-{test_bstate}_nfeats-{n_feat}"
        # filename = os.path.join(out_dir,f"{file_str}.npy")
        # np.save(filename, predictions_run)
        # print("...done")
        #- #%% =======================================================================================
          

RESULTS['prediction_results'] = pred_results

print("saving...")
file_str = f"prediction_results_overall_nfeats-{n_feat}"
filename = os.path.join(out_dir,f"{file_str}.npy")
np.save(filename, RESULTS)
print("...done")

#%% =======================================================================================

