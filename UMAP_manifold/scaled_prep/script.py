# Loading data
import time
import os
from matplotlib import pyplot as plt
import numpy as np
import umap
import scipy
import gc
from pathlib import Path
import pickle as pkl
import pandas as pd
import xarray as xr
from tqdm import tqdm
from scipy import ndimage
import math
from scipy.stats import pearsonr

def plot_raster(neural_data_fullUnlockAux, neural_data_BaselineTrialAux):
    plt.imshow(neural_data_fullUnlockAux, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()
    plt.imshow(neural_data_BaselineTrialAux, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()


def central_event_for_scaling(events, period):
    if period == 'scaled_prep':
        return events.unlock.values
    elif period == 'scaled_tone':
        return events.tone_onset.values
    elif period == 'full_unlock':
        return events.tone_onset.values
    else:
        raise ValueError(f"Invalid period for scaling: {period}")


def binned_spikes_and_rate_per_cell(events, spikes, period, single_trial=None, rescale=False):
    from manifolds.prepr import bin_spikes, spikes_for_trial_per_cell, get_window

    numTrials = len(events)
    dt = 1/fs

    neural_data = []
    time = []

    trial_window = get_window(events, baseline, period)
    cells = np.unique(spikes['cell_id'].values)
    cellDepths = spikes.attrs['cell_depths']

    num_spikes_per_cell = np.zeros(len(cells))

    if single_trial is None or single_trial == 'avg':
        trials = range(numTrials)
        duration = (trial_window[:, 1] - trial_window[:, 0]).sum()
    else:
        trials = [single_trial]
        duration = trial_window[single_trial, 1] - trial_window[single_trial, 0]

    if rescale:
        central_ev = central_event_for_scaling(events, period)
        desired_pre = (central_ev - trial_window[:, 0]).mean()
        desired_post = (trial_window[:, 1] - central_ev).mean()

        task_progress = np.r_[np.zeros(int(desired_pre / dt)), np.ones(int(desired_post / dt))]
    else:
        task_progress = None

    for trial in trials:

        wdw_start, wdw_end = trial_window[trial]
        spike_times = spikes_for_trial_per_cell(spikes, trial, cells)

        if rescale:
            if desired_pre == 0:
                pre_ratio = 1
            else:
                pre_ratio = (central_ev[trial] - wdw_start) / desired_pre

            if desired_post == 0:
                post_ratio = 1
            else:
                post_ratio = (wdw_end - central_ev[trial]) / desired_post

            # Normalize spike times to align and scale spikes across trials
            for spt in spike_times: # for each neuron
                spt -= central_ev[trial]
                spt[spt < 0] = spt[spt < 0] / pre_ratio
                spt[spt > 0] = spt[spt > 0] / post_ratio
                spt += central_ev[trial]

            # after rescaling, need to update the window
            wdw_start = central_ev[trial] - desired_pre
            wdw_end = central_ev[trial] + desired_post

        neural_data_for_trial, time_for_trial = bin_spikes(spike_times, dt, wdw_start, wdw_end)
        neural_data_for_trial = neural_data_for_trial.T

        num_spikes_per_cell += np.sum(neural_data_for_trial, axis=1)
        # if plot: plot_raster(neural_data_fullUnlockAux, neural_data_BaselineTrialAux)
        neural_data.append(neural_data_for_trial)
        time.append(time_for_trial)
    
    rate_per_cell = num_spikes_per_cell / duration

    if single_trial == 'avg':
        # length may fluctuate due to numerical error, but it's okay to clip
        min_len = min(arr.shape[1] for arr in neural_data)
        if task_progress is not None:
            min_len = min(min_len, task_progress.shape[0])
            task_progress = task_progress[:min_len]
        neural_data = np.array([arr[:, :min_len] for arr in neural_data])
        time = np.array([arr[:min_len] for arr in time])
        # now sum across trials
        neural_data = np.sum(neural_data, axis=0)
    elif single_trial is None:
        # concatenate all trials
        neural_data = np.hstack(neural_data)
        time = np.hstack(time)
    else:
        # TODO: implement for single trial
        pass

    return neural_data, time, rate_per_cell, task_progress, cellDepths

def overlapping_window(np_array, window_size=25):
    return ndimage.uniform_filter1d(np_array, size=window_size, axis=1, mode='constant')

def non_overlapping_window(np_array, window_size=25):
    window_hop = window_size
    start_frame = window_size
    end_frame = window_hop * math.floor(float(np_array.shape[1]) / window_hop)
    window = []
    for frame_idx in range(start_frame, end_frame, window_hop):
        window.append(np.mean(np_array[:, frame_idx - window_size:frame_idx],  axis=1)) # Add mean

    return np.transpose(np.vstack(window))

def UMAP(n_neighbors,min_dist,n_components,metric,randomNumber,Raster):
    umap_reduction = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                n_components=n_components,
                                metric=metric, random_state=randomNumber).fit(Raster.T)
    umap_representation = umap_reduction.transform(Raster.T)
    umap_representation_back = umap_reduction.inverse_transform(umap_representation).T
    (pearsonCorr, pvalue) = pearsonr(Raster.flatten(), umap_representation_back.flatten())

    return umap_representation, umap_reduction, pearsonCorr, pvalue

def plot_umap_representation(repr, names, n_components, task_progress=None, saveAt=None, show=False):

    # Calculate the number of rows and columns for the grid
    n_trials = len(repr)
    n_cols = min(4, n_trials)
    n_rows = (n_trials + n_cols - 1) // n_cols

    # Create a new figure with a grid of subplots
    if n_components == 2:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)
    elif n_components >= 3:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), subplot_kw={'projection': '3d'}, squeeze=False)
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    for i, umap_representation in enumerate(repr):
        ax = axes_flat[i]
        if task_progress[i] is None or len(np.unique(task_progress[i])) <= 1:
            colors = plt.cm.viridis(np.linspace(0, 1, len(umap_representation)))
        else:
            progress = task_progress[i]
            # make milder colors (now they are 0 and 1)
            progress[progress == 0] = 0.2
            progress[progress == 1] = 0.8
            colors = plt.cm.viridis(progress)
        if n_components == 2:
            ax.plot(umap_representation[:, 0], umap_representation[:, 1], color='gray', linewidth=0.5)
            scatter = ax.scatter(umap_representation[:, 0], umap_representation[:, 1], c=colors)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(f'Sess. {names[i]}')
        elif n_components >= 3:
            ax.plot(umap_representation[:, 0], umap_representation[:, 1], umap_representation[:, 2], color='gray', linewidth=0.5)
            scatter = ax.scatter(umap_representation[:, 0], umap_representation[:, 1], umap_representation[:, 2], 
                                 c=colors, marker='.')
            ax.set_xlabel('C1')
            ax.set_ylabel('C2')
            ax.set_zlabel('C3')
            ax.set_title(f'Sess. {names[i]}')
        else:
            ax.plot(umap_representation[:, 0])
            ax.set_xlabel('Time')
            ax.set_ylabel('Component 1')
            ax.set_title(f'Sess. {names[i]}')

    # Remove any unused subplots
    for j in range(i+1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    # fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Time')
    if saveAt is not None:
        plt.savefig(saveAt)

    if show:
        plt.show()

    return fig, axes

def save_umap_results(reprs, reds, names, task_progress, reg, n_components, period, validCells):
    import joblib

    results = {
        'representations': reprs,
        'reductions': reds, # Python 3.11 has an issue saving and loading this info. To use in Expanse, with Python3.11, need tor remove this 
        'folder_names': names,
        'task_progress': task_progress,
        'validCellsDepth': validCells
    }

    filename = filename_base(period, n_components, reg) + '.pkl'
    joblib.dump(results, filename)

    print(f"Saved UMAP results to {filename}")


def load_umap_results(reg, n_components, period):
    import pickle
    with open(f'{filename_base(period, n_components, reg)}.pkl', 'rb') as f:
        loaded_results = pickle.load(f)

    loaded_reprs = loaded_results['representations']
    loaded_reds = loaded_results['reductions']
    loaded_names = loaded_results['folder_names']
    task_progress = loaded_results['task_progress']
    return loaded_reprs, loaded_reds, loaded_names, task_progress

def process_umap(period, n_components, folders, reg):
    reprs, reds, names, task_progress, validCellsDepth = [], [], [], [], []
    from preprocess import create_task_events, epoch_data
    for folder in tqdm(folders):
        try:
            sess_dir_path = Path(f'./data/{folder}')
            # Load epoched data
            events = create_task_events.load_task_events(sess_dir_path)
            spikes = epoch_data.load_epoched_spikes(sess_dir_path, reg)
            # attrs = spikes.attrs
        except FileNotFoundError as e:
            print(f"Error parsing {folder}: {e}.\nSkipping to the next one..\n")
            continue

        # get baseline and trial firing rates to find valid cells (spiking above certain frequency)
        _, _, meanRateBaseline, _, _ = binned_spikes_and_rate_per_cell(events, spikes, 'baseline')
        _, _, meanRateTrial, _, _ = binned_spikes_and_rate_per_cell(events, spikes, 'trial')

        # Group spiking results for all trials
        neural_data, time, rate_per_cell, task_progr, cellDepths  = binned_spikes_and_rate_per_cell(events, spikes, period, single_trial='avg', rescale=True)

        # find cells spiking above certain frequency
        validCells = np.argwhere((meanRateBaseline >= minRateBaseline) * (meanRateTrial > minRateTrial)).flatten()
        neural_data = neural_data[validCells]

        # Get final neural trajectory with overlapping or non-overlapping window
        neural_traj = overlapping_window(neural_data, window_size=window_size)
        # neural_traj = non_overlapping_window(neural_data, window_size=window_size)
        # Convert to rate
        neural_traj /= bin_time

        # Calculate UMAP representation
        try:
            umap_representation, umap_reduction, pearsonCorr, pvalue = UMAP(n_neighbors, min_dist, n_components, metric, randomNumber, neural_traj)
            print(f"Pearson correlation: {pearsonCorr}")

            reprs.append(umap_representation)
            reds.append(umap_reduction._raw_data)
            names.append(folder)
            task_progress.append(task_progr)
            validCellsDepth.append(cellDepths[validCells])
        except Exception as e:
            print(f"Error calculating UMAP for {folder}: {e}")
            continue

    save_umap_results(reprs, reds, names, task_progress, reg, n_components, period, validCellsDepth)

    fig, axes = plot_umap_representation(reprs, names, n_components, task_progress, saveAt=f'{filename_base(period, n_components, reg)}.png')
    return fig, axes

def dir_base(period):
    return f'./manifolds/{period}'

def filename_base(period, n_components, reg):
    return f'{dir_base(period)}/umap_results_n{n_components}_{reg}'

if __name__ == '__main__':

    # Root folder containing sessions as subfolders
    dirpath_data = Path('./data')

    baseline = 1 # time before tone onset (in seconds), constant for all trials
    minRateBaseline, minRateTrial = 0.25, 0.5 # minimum firing rate for a cell to be considered
    bin_time = 0.05 # time bin in secs
    fs = 500
    window_size = int(bin_time*fs)
    n_neighbors, min_dist, metric, randomNumber = 50, 0.25, 'euclidean', 42

    paramsDict = {'baseline': baseline, 'minRateBaseline': minRateBaseline, 'minRateTrial': minRateTrial, 'bin_time': bin_time, 'fs': fs,
                  'window_size': window_size, 'n_neighbors': n_neighbors, 'min_dist': min_dist, 'metric': metric, 'randomNumber': randomNumber}

    import json
    with open("manifolds/UMAP_params.json", "w") as fp:
        json.dump(paramsDict , fp) 

    period = 'scaled_prep' # 'aligned_tone', 'aligned_prep', 'full_unlock', 'scaled_prep', 'scaled_tone'

    os.makedirs(dir_base(period), exist_ok=False)
    import shutil
    # Copy the script to the destination folder
    shutil.copy(Path(__file__), Path(dir_base(period)) / 'script.py')

    plot=False

    # List subfolders
    folders = [f.name for f in dirpath_data.iterdir() if f.is_dir()]

    for reg in ['m1']:
        for n_components in [2]:

            # # load and plot
            # reprs, reds, names, task_progress = load_umap_results(reg, n_components, period)
            # fig, axes = plot_umap_representation(reprs, names, n_components=n_components, task_progress=task_progress, saveAt=f'{filename_base(period, n_components, reg)}.png', show=True)
            
            # compute
            process_umap(period, n_components, folders, reg)

# 1) Find an "average low-dimensional representation" of the baseline and movement periods -> UMAP now, but CEBRA is better.
#         1.1) With UMAP: will mean trial-averaged manifold, for each session. Then we can compare session by session.
#         UMAP will give different number of samples per trial, and per session (we will need to interpolate manifold data)
#         1.2) With CEBRA, the number of outputs is the same, regardless of the session.
# 2) Use that representation to calibrate the M1 model