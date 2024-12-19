import os

nthreads = "32" # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import sklearn
import numpy as np
from pathlib import Path
import sklearn.model_selection
import pandas as pd
import combine_data
from eeg_research.system.bids_selector import BidsArchitecture, BidsDescriptor

def Main(task):
    study_directory = Path("/data2/Projects/eeg_fmri_natview/derivatives")
    paramters = {
        "root": study_directory,
        "datatype": "multimodal",
        "suffix": "multimodal",
        "description": "CustomGfpBk8",
        "run": "01",
        "task": task,
        "extension": ".pkl",
    }

    architecture = BidsArchitecture(**paramters)
    descriptor = architecture.report()
    
    caps = np.array(['CAP1',
                     'CAP2',
                     'CAP3',
                     'CAP4',
                     'CAP5',
                     'CAP6',
                     'CAP7',
                     'CAP8'])

    SAMPLING_RATE_HZ = 3.8
    WINDOW_LENGTH_SECONDS = 10


    r_data_for_df = {
                    'subject':[],
                    'session':[],
                    'ts_CAPS':[],
                    'pearson_r':[],
                    'frequency_Hz': [],
                    }
    
    
    big_d = combine_data.pick_data(architecture=architecture)
    
    for subject in descriptor.subjects:
        train_database, test_database = combine_data.generate_train_test_architectures(
                architecture = architecture,
                train_subjects = list(descriptor.subjects),
                test_subjects = subject,
            )
        
        for _, test_session in test_database.iterrows():
            train_keys, test_keys = combine_data.generate_train_test_keys(
                train_database = train_database,
                test_database = test_session
            )
            
            try:
                for band in range(39): 
                    feat_args = {"eyetracking":["pupil_dilation","first_derivative","second_derivative"],
                                "eeg":{
                                    "channel": None,
                                    "band": band,
                                },
                    }
                    for cap in caps:
                        print(f"===== {cap} =====")
                        X_train, Y_train, X_test, Y_test = combine_data.create_train_test_data(
                            big_data=big_d,
                            train_keys=train_keys,
                            test_keys= [test_keys],
                            cap_name = cap,
                            features_args=feat_args,
                            window_length=int(SAMPLING_RATE_HZ*WINDOW_LENGTH_SECONDS),
                            masking = True,
                            trim_args = (5,None),
                        )

                        
                        estimator = sklearn.linear_model.RidgeCV(cv=5, )
                        shapes = [
                            *X_train.shape,
                            *Y_train.shape,
                            *X_test.shape,
                            *Y_test.shape,
                        ]

                        if any([shape == 0 for shape in shapes]):
                            continue

                        estimator.fit(X_train,Y_train)
                        Y_hat = estimator.predict(X_test)
                        r = np.corrcoef(Y_test.T,Y_hat.T)[0,1]
                        for key, values in zip(
                            [
                                'subject',
                                'session',
                                'ts_CAPS',
                                'pearson_r',
                                'frequency_Hz',
                                ],
                            [
                                subject,
                                test_session['session'],
                                cap,
                                r,
                                band + 1,
                            ]):
                            r_data_for_df[key].append(values)
                
            except Exception as e:
                raise e

    df_pearson_r = pd.DataFrame(r_data_for_df)
    df_pearson_r.to_csv(f'prediction_pupil_and_gfp_{task}.csv')

if __name__ == "__main__":
    tasks = ['dme','dmh','inscapes','monkey1','monkey2','monkey5','peer','tp']
    for task in tasks:
        Main(task = task)