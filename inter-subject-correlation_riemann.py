import numpy as np
from scipy.linalg import eigh
from timeit import default_timer
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space, untangent_space

def regularize_covariance(matrix, gamma=0.1):
    """Regularize a covariance matrix using a scaled identity matrix."""
    return (1 - gamma) * matrix + gamma * np.mean(eigh(matrix)[0]) * np.identity(matrix.shape[0])

def train_cca(data):
    """Run Correlated Component Analysis on your training data with Riemannian mean covariance estimation."""

    start = default_timer()
    C = len(data.keys())
    print(f'train_cca - calculations started. There are {C} conditions')

    gamma = 0.1
    Rw_list, Rb_list = [], []

    for cond in data.values():
        N, D, T = cond.shape
        print(f'Condition has {N} subjects, {D} sensors and {T} samples')
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)

        # Calculate within-subject and between-subject covariance matrices
        Rw_list.extend([Rij[i, i, :, :] for i in range(N)])
        Rb_list.extend([Rij[i, j, :, :] for i in range(N) for j in range(N) if i != j])

    # Riemannian mean for Rw and Rb
    Rw = mean_riemann(Rw_list)
    Rb = mean_riemann(Rb_list)

    # Regularization of Rw
    Rw_reg = regularize_covariance(Rw, gamma=gamma)

    # Compute ISCs and W using generalized eigenvalue decomposition
    ISC, W = eigh(Rb, Rw_reg)
    ISC, W = ISC[::-1], W[:, ::-1]  # Sort in descending order

    stop = default_timer()
    print(f'Elapsed time: {round(stop - start)} seconds.')
    return W, ISC

def apply_cca(X, W, fs):
    """Apply precomputed spatial filters to your data with Riemannian mean covariance estimation."""

    start = default_timer()
    print('apply_cca - calculations started')

    subjects, channels, samples = X.shape
    window_sec = 5
    X = X.reshape(channels * subjects, samples)

    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (subjects, channels, subjects, channels)), 1, 2)

    # Calculate within-subject and between-subject covariance matrices using Riemannian mean
    Rw_list = [Rij[subject, subject, :, :] for subject in range(subjects)]
    Rb_list = [Rij[i, j, :, :] for i in range(subjects) for j in range(subjects) if i != j]

    Rw = mean_riemann(Rw_list)
    Rb = mean_riemann(Rb_list)

    # Compute ISCs
    ISC = np.sort(np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W))[::-1]

    # Scalp projections
    A = np.linalg.solve(Rw @ W, np.transpose(W) @ Rw @ W)

    # ISC by subject
    print('by subject is calculating')
    ISC_bysubject = np.empty((channels, subjects))

    for subj_k in range(subjects):
        subj_Rw_list = [Rij[subj_k, subj_k, :, :]] + [Rij[subj_l, subj_l, :, :] for subj_l in range(subjects) if subj_k != subj_l]
        subj_Rb_list = [Rij[subj_k, subj_l, :, :] for subj_l in range(subjects) if subj_k != subj_l]

        Rw = mean_riemann(subj_Rw_list)
        Rb = mean_riemann(subj_Rb_list)

        ISC_bysubject[:, subj_k] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)

    # ISC per second
    print('by persecond is calculating')
    ISC_persecond = np.empty((channels, int(samples / fs) + 1))
    window_i = 0

    for t in range(0, samples, fs):
        Xt = X[:, t:t + window_sec * fs]
        Rij = np.cov(Xt)

        # Rw and Rb for each time window using Riemannian mean
        window_Rw_list = [Rij[i:i + channels, i:i + channels] for i in range(0, channels * subjects, channels)]
        window_Rb_list = [Rij[i:i + channels, j:j + channels]
                          for i in range(0, channels * subjects, channels)
                          for j in range(0, channels * subjects, channels) if i != j]

        Rw = mean_riemann(window_Rw_list)
        Rb = mean_riemann(window_Rb_list)

        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
        window_i += 1

    stop = default_timer()
    print(f'Elapsed time: {round(stop - start)} seconds.')
    return ISC, ISC_persecond, ISC_bysubject, A
