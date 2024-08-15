
import os

nthreads = "32" # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import pickle
import combine_data

if __name__ == '__main__':
    study_folder = (
        '/data2/Projects/eeg_fmri_natview/derivatives'
        '/multimodal_prediction_models/data_prep'
        '/prediction_model_data_eeg_features_v2/dictionary_group_data_Hz-3.8'
    )

    data = combine_data.combine_data_from_filename(
        reading_dir = study_folder,
        task = 'checker',
        run = '01BlinksRemoved'
        )
    
    grid_search = combine_data.fine_tune_model(data)
    with open('./grid_search.pkl', 'wb') as f:
        pickle.dump(grid_search, f)

    