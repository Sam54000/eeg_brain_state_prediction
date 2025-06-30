#%%
import itertools
import matplotlib.pyplot as plt
import pickle
import mne
from itertools import pairwise
from typing import Callable
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.decoding import Scaler
from mne.decoding import Vectorizer
from mne.io.edf import read_raw_edf
import scipy
from mne.datasets import eegbci
import numpy as np
import pandas as pd
import scipy.stats
from scipy.interpolate import CubicSpline
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold 
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from mne.decoding import cross_val_multiscore
from sklearn.pipeline import Pipeline

import pyriemann
from pyriemann.estimation import Coherences, Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.spatialfilters import AJDC
from pyriemann.classification import SVC
from pyriemann.regression import SVR
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import TSclassifier
import bids_explorer.architecture.architecture as arch
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.base import nearest_sym_pos_def

class NearestSPD(TransformerMixin, BaseEstimator):
    """Transformer to convert matrices to their nearest symmetric positive definite version.
    
    This transformer uses the nearest_sym_pos_def function from pyriemann to ensure
    matrices are symmetric and positive definite.
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit the transformer.
        
        No fitting is needed as this is a stateless transformer.
        """
        return self

    def transform(self, X):
        """Transform the input matrices to their nearest SPD versions.
        
        Args:
            X (ndarray): Input matrices of shape (n_samples, n_features, n_features)
            
        Returns:
            ndarray: Transformed matrices of same shape as input
        """
        transformed = nearest_sym_pos_def(X)
        return transformed

class Connectivities(Coherences):
    """Getting connectivity features from epoch"""

    def transform(self, X):
        X_coh = super().transform(X)
        X_con = np.mean(X_coh, axis=-1, keepdims=False)
        return X_con


def zscore(raw: mne.io.Raw) -> mne.io.Raw:
    zscored = scipy.stats.zscore(
        raw.pick_type("eeg").get_data(), axis = 1
        )
    return zscored
    
def preprocess(raw: mne.io.Raw) -> mne.io.Raw:
    raw.resample(250)
    raw.filter(l_freq=1, h_freq=30)
    raw.pick_channels(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'])
    return raw

def epoching(raw: mne.io.Raw, times: np.ndarray, time_window: tuple) -> np.ndarray:
    """Preprocess single file to get epochs in numpy arrays.
    
    Creates TR events every 2.1 seconds starting from the first R128 event.
    Epochs are created around these TR events.
    
    Args:
        raw (mne.io.Raw): The raw mne object to process.
    
    Returns:
        numpy.ndarray: the epochs (n_epochs, n_channels, n_times)
    """
    event_tr = times * raw.info["sfreq"]
    events_tr = np.zeros((len(event_tr), 3), dtype=int)
    events_tr[:, 0] = event_tr
    events_tr[:, 2] = 999
    epochs = mne.Epochs(
        raw=raw,
        events=events_tr,
        tmin=time_window[0],
        tmax=time_window[1]
    )

    return epochs.get_data()

def labelize(brainstates: dict) -> np.ndarray:
    """Labelize the brainstates.
    
    Four classes are created from the brainstates time-series. brainstates
    classes (C1, C2, C3 and C4) are the mean of brainstates pairs:
    - C1 = mean(CAP1, CAP2)
    - C2 = mean(CAP3, CAP4)
    - C3 = mean(CAP5, CAP6)
    - C4 = mean(CAP7, CAP8)
    
    Then, for each time point we take the name of the brainstate 
    having the maximum value resulting in a vector of labels.
    
    Args:
        brainstates (dict): The brainstates that were pre-processed and saved
            in pickle format.
    
    Returns:
        numpy.ndarray: The labels for each time point.
    """
    indices = [0,2,4,6]
    classes = np.empty((4,brainstates["time"].shape[0]))
    for idx, pair in enumerate(indices):
        classes[idx,:] = np.mean(
            abs(brainstates["feature"][pair:pair+1,:]),
            axis=0
        )
    label_indices = np.argmax(classes, axis=0)
    return label_indices

def simple_experiment_regression():

    scaling = [
        ("No_Scaling", None),
        ("Zscore_scale",Scaler(scalings="mean")),
        ("Median_scale", Scaler(scalings="median"))
    ]
    filters = [
        ("No_Filter", None), 
        ("CSP", CSP(nfilter=4, metric="riemann", log = False)),
        ("SPoC", CSP(nfilter=4, metric="riemann", log = False)),
        ]
    regressors = [
        ("SVR", SVR(C=0.01)),
        ("KNearestNeighbor", pyriemann.regression.KNearestNeighborRegressor()),
        #("HGBR", HistGradientBoostingRegressor()),
        ]

    product = list(itertools.product(
        scaling,
        filters, 
        regressors,
        ))

    pipelines = []

    for p in product:
        steps = [
            ("Covariances", Covariances(estimator="lwf")),
        ]
        if p[0][1] is not None:
            steps.insert(0, p[0])
        for step in p[1:]:
            if step[1] is None:
                continue
            else:
                steps.append(step)
        pipelines.append(Pipeline(steps=steps))
    
    return pipelines

def simple_experiment_classification():
    scaling = [
        ("No_Scaling", None),
        ("Zscore_scale", Scaler(scalings="mean")),
        ("Median_scale", Scaler(scalings="median"))
    ]
    filters = [
        ("No_filter", None),
        ("CSP", CSP(nfilter=4, metric="riemann", log = False)),
        ]
    classifier = [
        ("SVC",SVC(C=0.01)), 
        ("TSclassifier", TSclassifier(metric="riemann")),
        #("HGBC", HistGradientBoostingClassifier()),
        ]

    product = list(itertools.product(
        scaling,
        filters, 
        classifier
        ))

    pipelines = []

    for p in product:
        steps = [
            ("Covariances", Covariances(estimator="lwf")),
        ]
        if p[0][1] is not None:
            steps.insert(0, p[0])
        for step in p[1:]:
            if step[1] is None:
                continue
            else:
                steps.append(step)
        pipelines.append(Pipeline(steps=steps))
    
    return pipelines

def csp_svm() -> Pipeline:
    param_svm = {"kernel": ("linear","rbf", "sigmoid", "poly"), 
                  "C": [0.01, 0.1, 1, 10, 100]}
    steps = [
        ("cov", Covariances(estimator="lwf")),
        ("csp", CSP(nfilter=4, metric="riemann")),
        ("optsvm", GridSearchCV(SVC(), param_svm, cv=5)),
    ]

    return Pipeline(steps=steps)

def feature_selection() -> Pipeline:
    step_fs = [
        ('cov', Covariances(estimator='lwf')),
        ('csp', CSP(n_components=4, metric='riemann')),
        ('ts', TangentSpace(metric='riemann')),
        ('select', SelectKBest(mutual_info_classif, k=20)),
        ('scaler', StandardScaler())
    ]
    return Pipeline(steps=step_csp)

def process_ensemble_func_con() -> Pipeline:

    spectral_met = ["cov", "lagged", "instantaneous"]
    fmin, fmax = 1, 30
    param_lr = {
        "penalty": "elasticnet",
        "l1_ratio": 0.15,
        "intercept_scaling": 1000.0,
        "solver": "saga",
    }
    param_ft = {"fmin": fmin, "fmax": fmax, "fs": 250}
    step_fc = [
        ("scale", Scaler(scalings="median")),
        ("spd", NearestSPD()),
        ("tg", TangentSpace(metric="riemann")),
        ("LogistReg", LogisticRegression(**param_lr)),
    ]
    ppl_fc = {}
    for sm in spectral_met:
        if sm == "cov":
            ppl_fc[sm] = Pipeline(
                steps=[("cov", Covariances(estimator="lwf"))] + step_fc
            )
        else:
            ft = Connectivities(**param_ft, coh=sm)
            ppl_fc[sm] = Pipeline(steps=[("ft", ft)] + step_fc)
    
    fc_estim = [(n, ppl_fc[n]) for n in ppl_fc]
    cvkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = LogisticRegression(**param_lr)
    ens = StackingClassifier(
        estimators=fc_estim,
        cv=cvkf,
        n_jobs=-1,
        final_estimator=lr,
        stack_method="predict_proba",
    )
    return ens
    
def single_brainstates_process(file: pd.Series):
    brainstates = pickle.load(open(file["filename"], "rb"))
    return labelize(brainstates), np.array(brainstates["time"])

def resample_brainstates(brainstates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Resample the brainstates to the times.
    
    Args:
        brainstates (np.ndarray): The brainstates to resample.
        times (np.ndarray): The times to resample to.
    """
    cs = CubicSpline(brainstates["time"], brainstates["feature"], axis = 1)
    return cs(times)

def generate_X_y_groups_epochs(
    selected_subjects: list[str],
    time_window: tuple,
    eeg_architecture: arch.BidsArchitecture,
    bs_architecture: arch.BidsArchitecture,
    classes: bool = True,
    nb_subjects: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate X, y, and groups for the classification.
    
    """
    X = []
    Y = []
    groups = []

    if nb_subjects is not None:
        selected_subjects = np.random.choice(
            eeg_architecture.subjects, 
            size=nb_subjects, 
            replace=False
        )
    else:
        selected_subjects = eeg_architecture.subjects

    for i, subject in enumerate(selected_subjects):
        subject_selection_eeg = eeg_architecture.select(subject = subject)
        subject_selection_bs = bs_architecture.select(subject = subject)

        for eeg_arch, bs_arch in zip(
            subject_selection_eeg,subject_selection_bs
            ):
            brainstates = pickle.load(open(bs_arch[1]["filename"], "rb"))

            if classes:
                bs = labelize(brainstates)
                bs_times = np.array(brainstates["time"])
            else:
                bs = brainstates["feature"]
                bs_times = np.array(brainstates["time"])

            raw = read_raw_edf(eeg_arch[1]["filename"], preload=True)
            epochs_data = epoching(preprocess(raw), bs_times, time_window)
            X.append(epochs_data*1e6)
            starting_index = int(np.ceil(abs(time_window[0]/2.1)))
            ending_index = int(bs_times.shape[0] - np.floor(abs(time_window[1]/2.1)))
            if bs.ndim == 1:
                bs = bs[starting_index:ending_index]
            else:
                bs = bs[:, starting_index:ending_index]
            Y.append(bs)
            groups.append(np.ones_like(bs) * i)

    X = np.concatenate(X, axis = 0)
    if Y[0].ndim == 1:
        Y = np.concatenate(Y, axis = 0)
    else:
        Y = np.concatenate(Y, axis = 1)
    groups = np.concatenate(groups, axis = 0)
    return X, Y, groups

#%%

#%%
def classification(eeg_architecture, bs_architecture):
    pipelines = simple_experiment_classification()
    pipelines.append(process_ensemble_func_con())

    # pipeline = csp_svm()
    #pipeline = process_ensemble_func_con()
        
    X, Y, groups = generate_X_y_groups_epochs(
        selected_subjects = eeg_architecture.subjects,
        time_window = (-10, 0),
        eeg_architecture = eeg_architecture,
        bs_architecture = bs_architecture,
        classes = True,
        nb_subjects = None,
    )
    df = pd.DataFrame()
    for pipeline in pipelines:
        if isinstance(pipeline, StackingClassifier):
            names = pipeline.estimators_
        else:
            names = list(pipeline.named_steps.keys())
        print(names)
        scores = cross_val_multiscore(
            pipeline,
            X,
            Y,
            groups = groups,
            cv= model_selection.LeaveOneGroupOut(),
            n_jobs=-1,
            scoring="accuracy",
            )
        sub_df = pd.DataFrame({
            "subjects": eeg_architecture.subjects,
            "Accuracy_scores": scores
        })
        sub_df["pipeline"] = "-".join(names)
        df = pd.concat([df,sub_df])
    df.to_csv("classifications_result.csv")

def regression(eeg_architecture, bs_architecture):
    X, Y, groups = generate_X_y_groups_epochs(
        selected_subjects = eeg_architecture.subjects,
        time_window = (-10, 0),
        eeg_architecture = eeg_architecture,
        bs_architecture = bs_architecture,
        classes = False,
        nb_subjects = None,
    )
    pipelines = simple_experiment_regression()
    df = pd.DataFrame()
    for pipeline in pipelines:
        if isinstance(pipeline, StackingClassifier):
            names = pipeline.estimators_
        else:
            names = list(pipeline.named_steps.keys())
        print(names)
        for cap in range(Y.shape[0]):
            scores = cross_val_multiscore(
                pipeline,
                X,
                Y[cap,:],
                groups = groups,
                cv= model_selection.LeaveOneGroupOut(),
                n_jobs=-1,
                scoring="r2",
                )
            sub_df = pd.DataFrame({
                "subjects": eeg_architecture.subjects,
                "R2_scores": scores,
                "ts_CAPS": cap + 1,
            })
            sub_df["pipeline"] = "-".join(names)
            df = pd.concat([df,sub_df])
    df.to_csv("regression_result.csv")

if __name__ == "__main__":
    eeg_architecture = arch.BidsArchitecture(
        root = "/data2/Projects/eeg_fmri_natview/chang_data/raw/",
    )
    bs_architecture = arch.BidsArchitecture(
        root = "/data2/Projects/eeg_fmri_natview/chang_data/derivatives/",
        datatype = "brainstates",
        extension = "pkl"
    )
    classification(eeg_architecture,bs_architecture)
    regression(eeg_architecture,bs_architecture)
