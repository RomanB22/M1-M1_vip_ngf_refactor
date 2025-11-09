import numpy as np
import scipy.signal
from tqdm import tqdm
import pickle, os
import matplotlib.pyplot as plt
import __main__

def bin_spikes(spike_times, dt, wdw_start, wdw_end):
    # Function that puts spikes into bins. Expects spike_times to be array of shape (n_neurons, n_spikes). Returns the binned data and the edges of the bins (left edge).

    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    num_neurons=spike_times.shape[0] #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data, edges[:-1]


def spikes_for_trial_per_cell(spikes, trial, cells):
    sp = [spikes.loc[(spikes['cell_id']==neuron) & 
                     (spikes['trial']==trial)].spike_time.values
                     for neuron in cells]
    return np.array(sp, dtype=object)


def get_window(events, baseline, period):
    # assert period in ['full_unlock', 'baseline_trial', 'baseline', 'trial', 'aligned'], \
    #     "Invalid window type, use one of the following: full_unlock, baseline_trial, baseline, trial, aligned"
    unlock = events.unlock.values
    tone_onset = events.tone_onset.values
    tone_offset = events.tone_offset.values

    return {
        'full_unlock': np.column_stack([unlock, tone_offset]),
        'baseline_trial': np.column_stack([tone_onset - baseline, tone_offset]),
        'baseline': np.column_stack([tone_onset - baseline, tone_onset]),
        'trial': np.column_stack([tone_onset, tone_offset]),
        'aligned_prep': np.column_stack([unlock, unlock + (tone_onset - unlock).max()]),
        'aligned_tone': np.column_stack([tone_onset, tone_onset + (tone_offset - tone_onset).max()]),
        'scaled_prep': np.column_stack([unlock, tone_onset]),
        'scaled_tone': np.column_stack([tone_onset, tone_offset])
    }[period]


def prepare_joint_session_data(session_id, task_events, epoched_spikes, smooth_spikes: bool, epoched_data, epoched_spikes_thal = None):
    """
    Prepare data for joint behavioral and neural analysis (CEBRA).
    """
    from manifolds.prepr import bin_spikes, spikes_for_trial_per_cell

    pmp = __main__.preprMetaParams

    use_lfp = pmp.val('rec_type') == 'lfp'
    subsample_lfp = True
    if use_lfp:
        lfp = epoched_data.LFP

    if epoched_spikes_thal is not None:
        cells_thal = np.unique(epoched_spikes_thal['cell_id'].values)
        if len(cells_thal) == 0:
            # if contains no spikes (just metadata), reset to None
            epoched_spikes_thal = None
        else:
            use_thal_as = pmp.val('use_thal_as')

    joystick = epoched_data.Joystick

    N = 10 # ensure dt is integer multiplier of joystick dt, otherwise need to replace slicing with interpolation
    dt = N * (joystick.time[1] - joystick.time[0]).item()

    dictParams = {'session_id': session_id,'dt': np.round(dt*1000,2), 'cell_depths': epoched_spikes.attrs['cell_depths'].tolist()}

    import json

    os.makedirs('CEBRA_params', exist_ok=True)
    with open(f'CEBRA_params/{session_id}_CEBRAparams.txt', 'w') as fp:
        json.dump(dictParams , fp) 

    cells = np.unique(epoched_spikes['cell_id'].values)

    central_event = epoched_spikes.attrs['central_event']
    trial_wind_rel = epoched_spikes.attrs['trial_window'] # relative to central_event

    all_neural_data = []
    all_aux = []
    all_trial_idx = []

    if use_lfp and subsample_lfp:
        num = len(epoched_spikes.attrs['cell_depths'])
        rng = np.random.default_rng(1234)
        subs_channels = rng.choice(lfp.chan.values, num, replace=False)
        lfp = lfp.sel(chan=subs_channels)

    for index, trial in task_events.iterrows():
        wdw_start, wdw_end = trial[central_event] + trial_wind_rel # now it's full trial. May also be narrower.
        # wdw_start, wdw_end = trial.unlock, trial.tone_offset

        def smooth_and_subsample(signal, N):
            kernel = np.ones(N) / N
            kernel = np.tile(kernel, (signal.shape[0], 1))
            smoothed = scipy.signal.fftconvolve(signal, kernel, mode='valid', axes=1)
            # subsample and crop to match stage bins
            signal = smoothed[:,::N]
            return signal

        if use_lfp:
            lfp_data = lfp.sel(trial=index,
                            time=slice(trial_wind_rel[0], trial_wind_rel[1]))
            neural_data = smooth_and_subsample(lfp_data, N)

            neural_data_time = np.arange(wdw_start, wdw_end, dt)[:neural_data.shape[1]]
        else:
            spikes = spikes_for_trial_per_cell(epoched_spikes, index, cells)
            neural_data, neural_data_time = bin_spikes(spikes, dt, wdw_start, wdw_end)
            neural_data = neural_data.T
        trial_id = np.full((neural_data.shape[1],), index)

        if wdw_start<trial.unlock and wdw_end>trial.tone_offset:
            stage_bins = [wdw_start, trial.unlock, trial.tone_onset, trial.tone_offset, wdw_end]
            # get stage at each time point of neural_data (0 - before, 1 - preparation, 2 - movement, 3 - after)
        else:
            if wdw_end>trial.tone_offset:
                stage_bins = [wdw_start, trial.tone_onset, trial.tone_offset, wdw_end]
            else:
                stage_bins = [wdw_start, trial.tone_onset, wdw_end]
        # print(np.unique(stage_bins))
        stage = np.digitize(neural_data_time, stage_bins) - 1 # -1 beacuse of how digitize returns bin indices
        stage = stage.astype(float) # as int is not supported by CEBRA


        try:
            signal = joystick.sel(trial=index,
                                    time=slice(trial_wind_rel[0], trial_wind_rel[1]))
            signal = smooth_and_subsample(signal, N)
            # crop to match stage bins
            signal = signal[:,:len(stage)]
        except:
            continue

        if epoched_spikes_thal is None:
            thal_spikes_as_aux = None
            # print(np.unique(stage) , len(signal), trial_wind_rel)
            aux = np.vstack((stage, signal))
        else:
            thal_spikes = spikes_for_trial_per_cell(epoched_spikes_thal, index, cells_thal)
            thal_spikes_as_aux, _ = bin_spikes(thal_spikes, dt, wdw_start, wdw_end)
            thal_spikes_as_aux = (thal_spikes_as_aux.T)[:,:len(stage)]
            # Zero pad thalamus spikes to match stage length if needed
            if thal_spikes_as_aux.shape[1] < len(stage):
                print("Warning. Thalamus spikes are shorter than stage. Padding with zeros.")
                pad_width = ((0, 0), (0, len(stage) - thal_spikes_as_aux.shape[1]))
                thal_spikes_as_aux = np.pad(thal_spikes_as_aux, pad_width, mode='constant', constant_values=0)

            aux = np.vstack((stage, signal, thal_spikes_as_aux))
        all_trial_idx.append(trial_id)
        all_neural_data.append(neural_data)
        all_aux.append(aux)

    neural_data = np.hstack(all_neural_data)
    all_aux = np.hstack(all_aux)
    all_trial_idx = np.hstack(all_trial_idx)

    with open(f'CEBRA_params/{session_id}_trial_idx.txt', 'w') as fp:
        json.dump(all_trial_idx.tolist() , fp)   

    if smooth_spikes:
        neural_data_smooth = smoothen_spikes(neural_data)
        # plot_data(neural_data, neural_data_smooth, all_aux)
        neural_data = neural_data_smooth

        if thal_spikes_as_aux is not None:
            thal_spikes_as_aux = smoothen_spikes(all_aux[3:,:])
            all_aux[3:,:] = thal_spikes_as_aux
    else:
        # plot_data(neural_data, auxiliary=all_aux)
        pass

    if epoched_spikes_thal is not None:
        if use_thal_as.startswith('th-pca'):
            pcaDims = int(use_thal_as[6:]) # drop 'th-pca' to stay with num dims
            from joystick.utils import doPCA

            thal_spikes_as_aux, _, explained_variance = doPCA(thal_spikes_as_aux.T, pcaDims)
            thal_spikes_as_aux = thal_spikes_as_aux.T
            print(f"Reduced thalamus spikes (aux) to {pcaDims} dimensions. Explained variance: {explained_variance[:pcaDims].sum()}")

            # put back to all_aux, removing dimesions reduced by PCA
            all_aux = all_aux[:3+pcaDims,:]
            all_aux[3:,:] = thal_spikes_as_aux
        elif use_thal_as == 'th-cebra':
            # load cebra model
            import cebra
            from pathlib import Path
            # TODO: this is hardcoded, need to link to proper model
            file_path = Path('output') / 'manifolds' / 'cebra' / 'neural_data_thal_spikes_smooth' / '230517_2759_1606VAL' / 'decode_rs_iter10000_ndim3_conddelta' / 'models' / 'model_time.pt'
            assert False, f"Need to fix this hardcoded path to CEBRA model for Thalamus. {file_path}"

            cebra_model = cebra.CEBRA.load(file_path)
            thal_spikes_as_aux = cebra_model.transform(thal_spikes_as_aux.T).T
            all_aux = all_aux[:3+cebra_model.output_dimension,:]
            all_aux[3:,:] = thal_spikes_as_aux

    return neural_data.T, all_aux.T # shape (time, num_neurons) and (time, aux_dim) where aux_dim is 3 (task stage, joystick_x, joystick_y)


def plot_data(session_id, neural_data, neural_data_smooth=None, auxiliary=None):
    # Plotting the neural data and smoothed data
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original neural data
    axes[0].plot(neural_data.T)
    # axes[0].imshow(neural_data, aspect='auto', cmap='viridis')
    axes[0].set_title('Original Neural Data')
    axes[0].set_ylabel('Neurons')
    axes[0].set_xlabel('Time')

    if neural_data_smooth is not None:
        # Plot smoothed neural data
        axes[1].plot(neural_data_smooth.T)
        # axes[1].imshow(neural_data_smooth, aspect='auto', cmap='viridis')
        axes[1].set_title('Smoothed Neural Data')
        axes[1].set_ylabel('Neurons')
        axes[1].set_xlabel('Time')

    plt.suptitle(session_id)
    plt.tight_layout()
    plt.show()


def smoothen_spikes(neural_data, kernel_size=23, sigma=2.5): # kernel_size should be an odd number
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize to preserve unit area
    
    # Repeat the kernel to match the number of neurons (M)
    kernel = np.tile(kernel, (neural_data.shape[0], 1))
    
    conv = scipy.signal.fftconvolve(neural_data, kernel, mode="same", axes=1)
    # conv = scipy.signal.fftconvolve(neural_data, kernel, mode="same") # wrong way of smoothing, but worked good - need to find why
    
    return conv

def preprocess_joint_data(raw_data_dir, save_as, smooth_spikes: bool):
    from preprocess.create_task_events import load_task_events
    from preprocess.epoch_data import load_epoched_spikes, load_epoched_data

    pmp = __main__.preprMetaParams

    # List subfolders
    folders = sorted([f.name for f in raw_data_dir.iterdir() if f.is_dir()])
    # folders = ['230517_2759_1606VAL'] # for testing

    neural_data = []
    auxiliary = []
    session_labels = []
    
    for folder in tqdm(folders):
        try:
            task_events = load_task_events(raw_data_dir / folder)
        except FileNotFoundError as e:
            print(f"Error parsing {folder}: {e}.\nSkipping to the next one..\n")
            continue

        path = raw_data_dir / folder
        for region in [pmp.val('region')]: # 'm1' or 'th'
            try:
                # Load epoched spikes (if needed)
                epoched_spikes = load_epoched_spikes(path, region)

                # optionally, with m1 data, can use thal as auxiliary data
                if  region == 'm1' and pmp.get('use_thal_as'):
                    epoched_spikes_thal = load_epoched_spikes(path, 'th')
                else:
                    epoched_spikes_thal = None

                data = load_epoched_data(path, region)
                neural, aux = prepare_joint_session_data(folder, task_events, epoched_spikes, smooth_spikes, data, epoched_spikes_thal)
                neural_data.append(neural)
                auxiliary.append(aux)
                session_labels.append(folder)
            except FileNotFoundError as e:
                print(f"Error parsing {region} data in {folder}: {e}.\nSkipping to the next one..\n")

    all_data = {'neural': neural_data, 'auxiliary': auxiliary, 'session_labels': session_labels}

    # Ensure the directory for save_as exists
    save_dir = os.path.dirname(save_as)
    os.makedirs(save_dir, exist_ok=True)

    with open(save_as, 'wb') as f:
        pickle.dump(all_data, f)

    return all_data


def load_preprocessed_joint_data(output_dir):

    all_data = pickle.load(open(output_dir, 'rb'))
    neural = all_data['neural']
    auxiliary = all_data['auxiliary'] 
    session_labels = all_data['session_labels']
    return neural, auxiliary, session_labels
