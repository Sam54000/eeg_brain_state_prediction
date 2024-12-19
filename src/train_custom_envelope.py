import os

nthreads = "1" # 64 on synapse
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
import argparse
import combine_data
from eeg_research.system.bids_selector import BidsArchitecture, BidsDescriptor


def Main(args):
    study_directory = Path("/data2/Projects/eeg_fmri_natview/derivatives")
    paramters = {
        "root": study_directory,
        "datatype": "multimodal",
        "suffix": "multimodal",
        "description": "CustomEnvBk8",
        "run": "01",
        "task": str(args.task),
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
                    'electrode': [],
                    }
    
    
    big_d = combine_data.pick_data(architecture=architecture)
    
    train_database, test_database = combine_data.generate_train_test_architectures(
            architecture = architecture,
            train_subjects = list(descriptor.subjects),
            test_subjects = args.subject,
        )

    for _, test_session in test_database.iterrows():
        train_keys, test_keys = combine_data.generate_train_test_keys(
            train_database = train_database,
            test_database = test_session
        )
        for band in range(39):
            for channel in range(61):
                feat_args = {"eyetracking":["pupil_dilation","first_derivative","second_derivative"],
                            "eeg":{
                                "channel": channel,
                                "band": band,
                            },
                }
                for cap in caps:
                    try:
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
                                'electrode',
                                ],
                            [
                                args.subject,
                                test_session['session'],
                                cap,
                                r,
                                band + 1,
                                channel,
                            ]):
                                r_data_for_df[key].append(values)
                
                    except Exception as e:
                        raise e

    df_pearson_r = pd.DataFrame(r_data_for_df)
    df_pearson_r.to_csv(f'/home/slouviot/01_projects/eeg_brain_state_prediction/data/sub-{args.subject}_task-{args.task}_desc-CustomEnvBk_predictions.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train_custom_envelope',)
    parser.add_argument('--subject', default='01')
    parser.add_argument('--task', default = "checker")
    args = parser.parse_args()
    Main(args)