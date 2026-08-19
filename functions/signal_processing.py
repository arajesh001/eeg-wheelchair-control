# =============================================================================
# IMPORTS
# =============================================================================

import mne
import numpy as np

# =============================================================================
# LOADING & SETUP
# =============================================================================

def load_raw_gdf (path:str, eog_channels:list[int]) -> tuple[mne.io.Raw, mne.io.Raw]:
    """
    Imports a gdf file using mne and returns only eeg channels.

    Args:
    path: string of the file path
    eog_channels: list of the indices of the eog channels

    Returns:
    A loaded file as an mne object.
    """
    raw = mne.io.read_raw_gdf(path, preload=True, eog=eog_channels)
    return raw


def rename_and_montage(raw:mne.io.Raw)->None:
    """
    Sets the montage to standard_1020 and renames the channel labels to match this format

    Args:
    raw: mne.io.Raw type that needs to be renamed and montaged (must be 22 eeg channels)

    Returns:
    None
    
    """

    ch_names_correct = [
    'Fz',
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
    ]

    rename_map = {old: new for old, new in zip(raw.ch_names, ch_names_correct)}
    raw.rename_channels(rename_map)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    return


# =============================================================================
# PREPROCESSING
# =============================================================================

def bandpass_filter(raw:mne.io.Raw, low, high, verbose:bool)->None:
    """
    Bandpasses the data to the frequency range [low, high] using a order 5 IIR butterworth filter.

    Args:
    raw: mne.io.Raw type that needs to be bandpassed
    low: lower bound frequency (Hz)
    high: upper bound frequency (Hz)
    verbose: bool which controls if outputs are printed

    Returns:
    None

    """
    raw.filter(l_freq=low, h_freq=high, method='iir',
               iir_params=dict(order=5, ftype='butter'), verbose=verbose)
    return

def run_ica(raw:mne.io.Raw, n_components:int)->mne.preprocessing.ICA:
    """
    Returns a fast ICA object of n_componenets of the data.

    Args:
    raw: mne.io.Raw type that needs to be ICA'd
    n_componenets: how many independent sources there are 

    Returns:
    Fitted mne.preprocessing.ICA object
    
    """
    ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(raw, picks='eeg')
    return ica

def find_eog_artifacts(ica:mne.preprocessing.ICA, raw:mne.io.Raw)->list[int]:
    '''
    Finds channel idx's correlated with eog channels.

    Args:
    ica: ica object to fit to raw
    raw: mne.io.Raw type that needs to be ICA'd

    Returns: list of bad channel idx's

    '''
    eog_indices, _ = ica.find_bads_eog(raw)
    return eog_indices

def apply_ica(ica:mne.preprocessing.ICA, raw:mne.io.Raw, exclude:list[int])->mne.io.Raw:
    """
    Fits ICA into a raw object and excludes the chosen componenets. 

    Args:
    ica: ica object to fit to raw
    raw: mne.io.Raw type that needs to be ICA'd
    exclude: a list of componenet indices to exclude

    Returns:
    mne.io.Raw object copy with ICA applied and chosen components excluded
    
    """
    ica.exclude = exclude
    raw_clean = ica.apply(raw.copy())
    return raw_clean


def get_motor_events(raw: mne.io.Raw) -> tuple[np.ndarray, dict]:
    """
    Extracts motor imagery events from a cleaned Raw object.

    Args:
        raw: cleaned mne.io.Raw object (post-ICA)

    Returns:
        events: np.ndarray of shape [n_events, 3]
        motor_event_id: dict mapping motor imagery class names to event codes
    """
    events, event_id = mne.events_from_annotations(raw)
    motor_event_id = {
        'left_hand':  event_id['769'],
        'right_hand': event_id['770'],
        'feet':       event_id['771'],
        'tongue':     event_id['772'],
    }
    return events, motor_event_id

# =============================================================================
# EPOCHING
# =============================================================================

def epoch_raw(raw: mne.io.Raw, events: np.ndarray, event_id: dict,
              tmin: float = 0.5, tmax: float = 4.0) -> mne.Epochs:
    """
    Extracts epochs around motor imagery events.

    Args:
        raw: cleaned mne.io.Raw object (post-ICA)
        events: events array from mne.events_from_annotations()
        event_id: dict mapping class names to event codes
        tmin: epoch start relative to event onset (seconds)
        tmax: epoch end relative to event onset (seconds)

    Returns:
        mne.Epochs object with annotations-based rejection applied
    """
    epochs = mne.Epochs(
        raw, events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True
    )
    return epochs



# USE THIS FOR SPECTOGRAM CNN
def compute_tfr(epochs: mne.Epochs,
                fmin: float = 8.0,
                fmax: float = 30.0,
                n_freqs: int = 65,
                tmin_crop: float = 1.0,
                tmax_crop: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Morlet wavelet TFR power and crops edge artifacts.

    Args:
        epochs: mne.Epochs object [n_epochs, 22, n_times]
        fmin: lower frequency bound (Hz)
        fmax: upper frequency bound (Hz)
        n_freqs: number of frequency bins
        tmin_crop: crop start after TFR (seconds, relative to event)
        tmax_crop: crop end after TFR (seconds, relative to event)

    Returns:
        power: np.ndarray [n_epochs, 22, n_freqs, n_times_cropped]
        labels: np.ndarray [n_epochs] of integer class labels
    """
    freqs = np.linspace(fmin, fmax, n_freqs)
    n_cycles = freqs / 2.0

    # [n_epochs, 22, n_freqs, n_times]
    power = mne.time_frequency.tfr_array_morlet(
        epochs.get_data(),
        sfreq=epochs.info['sfreq'],
        freqs=freqs,
        n_cycles=n_cycles,
        output='power'
    )

    # crop edge artifacts along time axis
    times = epochs.times
    time_mask = (times >= tmin_crop) & (times <= tmax_crop)
    power = power[:, :, :, time_mask]

    # integer labels: left=0, right=1, feet=2, tongue=3
    # Map raw event codes -> string names -> clean 0-3 labels
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
    
    raw_ids = epochs.events[:, 2]
    labels = np.array([label_map[id_to_name[raw_id]] for raw_id in raw_ids])

    return power, labels


# USE THIS FOR EEG CNN
def extract_raw_epochs(epochs: mne.Epochs,
                       tmin_crop: float = 0.5,
                        tmax_crop: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts raw EEG time series data from an Epochs object w/o
    time-freq transformation.

    Args:
        epochs: mne.Epochs object [n_epochs, 22, n_timepoints]
        tmin_crop: crop start 
        tmax_crop: crop end 

    Returns:
        data:np.ndarray [n_epochs, 22, n_timepoints] raw voltage values
        labels: np.ndarray [n_epochs] integer class labels (0-3)
    """

    # shape should be [n_epochs, 22, n_timepoints]
    data = epochs.get_data()

    # crop edge artifacts along time axis
    times = epochs.times
    time_mask = (times >= tmin_crop) & (times <= tmax_crop)
    data = data[:, :, time_mask]

    # Map raw event codes -> string names -> clean 0-3 labels
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
    
    raw_ids = epochs.events[:, 2]
    labels = np.array([label_map[id_to_name[raw_id]] for raw_id in raw_ids])

    return data, labels

