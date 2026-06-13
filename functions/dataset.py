# =============================================================================
# IMPORTS
# =============================================================================

import mne
import numpy as np

import sys
from pathlib import Path
from signal_processing import load_raw_gdf, rename_and_montage, bandpass_filter, run_ica, apply_ica, epoch_raw, compute_tfr, get_motor_events

# =============================================================================
# FUNCTIONS
# =============================================================================

def log_normalize(power: np.ndarray) -> np.ndarray:
    """
    Applies per triplet log normalization to a power array. 

    Args:
    power: np.ndarray of powers
    
    Returns:
    Same array but log normalized. 
    """
    
    epsilon = 1 * 10**(-10)
    log_power = np.log(power + epsilon)
    mean = log_power.mean(axis=-1, keepdims=True)
    std = log_power.std(axis=-1, keepdims=True)
    return (log_power - mean) / (std + epsilon)


def load_subject(subject_id: int, data_dir: str = "BCICIV_2a_gdf") -> tuple[np.ndarray, np.ndarray]:
    """
    Applies full signal processing pipeline (from pipeline.ipynb) to any subject
    in the dataset that is of a "T" (training) file type.

    Args:
    subject_id: which of the 9 subjects we need/ 
    data_dir: default dir is BCICIV_2a_gdf. 
    """

    # make filename
    filename = f"A0{subject_id}T.gdf"

    # import data, drop EOG channels and keep only EEG
    gdf_path = Path.cwd().parent / "data" / data_dir / filename
    raw = load_raw_gdf(gdf_path, [22, 23, 24])

    # montage channels + rename to the standard standard_1020  
    rename_and_montage(raw)

    # bandpass from 3-35 Hz
    bandpass_filter(raw, 3, 35, True)

    # run ica
    ica = run_ica(raw, 20)

    # IMPORTANT: apply_ica W/O dropping any componenets -> optimize later 
    # after pipeline works since it requires manual inspection
    raw_clean = apply_ica(ica, raw, [])

    # obtaining events and ids 
    events, event_id = get_motor_events(raw_clean)

    # epoching the data
    raw_epochs = epoch_raw(raw_clean, events, event_id)

    # obtaining power and labels using tfr func
    power, labels = compute_tfr(raw_epochs)

    # final tensor log normalized 
    final_tensor = log_normalize(power)

    return final_tensor, labels

def build_dataset(subject_ids: list[int], data_dir: str = "BCICIV_2a_gdf") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the full preprocessing pipeline across all specified subjects &
    concatenates into single dataset. Tracks subject identity per
    epoch to support LOSO cross validation.

    Args:
        subject_ids: list of subject ids to include
        data_dir: default BCICIV_2a_gdf

    Returns:
        X: np.ndarray [total_epochs, 22, n_freqs, n_times]
        y: np.ndarray [total_epochs] integer class labels
        subjects: np.ndarray [total_epochs] subject ID/epoch
    """
    X_list = []
    y_list = []
    subject_list = []

    for subject_id in subject_ids:
        power, labels = load_subject(subject_id, data_dir=)
        X_list.append(power)
        y_list.append(labels)
        subject_list.extend([subject_id] * len(power))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subjects = np.array(subject_list)

    return X, y, subjects


def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> tuple:
    pass