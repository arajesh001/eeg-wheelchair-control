# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    final_tensor = log_normalize(power).astype(np.float32)

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
        power, labels = load_subject(subject_id, data_dir)
        X_list.append(power)
        y_list.append(labels)
        subject_list.extend([subject_id] * len(power))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subjects = np.array(subject_list)

    return X, y, subjects


def save_dataset(X, y, subjects, save_dir: str) -> None:
    """
    Saves the dataset made by build_dataset

    Args:
    X: np.ndarray [total_epochs, 22, n_freqs, n_times]
    y: np.ndarray [total_epochs] integer class labels
    subjects: np.ndarray [total_epochs] subject ID/epoch
    save_dir: directory to save in 

    Returns:
    None

    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    np.save(save_path / "X.npy", X)
    np.save(save_path / "y.npy", y)
    np.save(save_path / "subjects.npy", subjects)

def load_dataset(save_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Loads the saved dataset

    Args:
    save_dir: directory with saved processed data

    Returns:
    X: np.ndarray [total_epochs, 22, n_freqs, n_times]
    y: np.ndarray [total_epochs] integer class labels
    subjects: np.ndarray [total_epochs] subject ID/epoch
    """
    
    save_path = Path(save_dir)
    X = np.load(save_path / "X.npy")
    y = np.load(save_path / "y.npy")
    subjects = np.load(save_path / "subjects.npy")
    return X, y, subjects

def get_loso_split(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                   test_subject_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into train and test sets using leave-one-subject-out.
    All epochs from test_subject_id go to test, all others go to train.

    Args:
        X: np.ndarray [total_epochs, 22, n_freqs, n_times]
        y: np.ndarray [total_epochs] integer class labels
        subjects: np.ndarray [total_epochs] subject ID/epoch
        test_subject_id: subject to leave out

    Returns:
        X_train, X_test, y_train, y_test -> np.ndarrays
    """

    # bool masking -> if == test_subject_id, it goes in test
    test_mask  = subjects == test_subject_id
    train_mask = subjects != test_subject_id

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test

def build_and_save_dataset(subject_ids: list[int], 
                           save_dir: str,
                           data_dir: str = "BCICIV_2a_gdf") -> None:
    """
    Runs the full preprocessing pipeline per subject and saves each one
    individually to disk immediately after processing. Never accumulates
    the full dataset in RAM —> peak memory is one subject at a time.

    Args:
        subject_ids: list of subject ids to process (1-9)
        save_dir: directory to save /subject files
        data_dir: directory w/ raw GDF files

    Returns:
        None
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    for subject_id in subject_ids:
        print(f"processing subject {subject_id}...")
        power, labels = load_subject(subject_id, data_dir)
        np.save(save_path / f"subject_{subject_id}_X.npy", power)
        np.save(save_path / f"subject_{subject_id}_y.npy", labels)
        print(f"subject {subject_id} saved -> {power.shape}")

# modifying dataloader so that it's one subject at a time -> make it easier on RAM
# original concatenated np array was too much for Mac -> ~10 GB at a time
class EEGDataset(torch.utils.data.Dataset):
    """
    Memory-efficient EEG dataset that loads per-subject .npy files via
    memory mapping rather than holding the full dataset in RAM at once.
    """
    def __init__(self, subject_ids: list[int], data_dir: str):
        self.subject_ids = subject_ids
        self.data_dir = data_dir
        self.X_list = []
        self.y_list = []

        save_path = Path(data_dir)

        for subject_id in subject_ids:
            # READS FROM DISK RATHER THAN ALL AT ONCE
            X = np.load(save_path / f"subject_{subject_id}_X.npy",
                        mmap_mode="r")
            y = np.load(save_path / f"subject_{subject_id}_y.npy",
                        mmap_mode="r")

            self.X_list.append(X)
            self.y_list.append(y)

        # cumulative epoch counts per subject
        self.cumulative_sizes = np.cumsum([len(x) for x in self.X_list])
        self.total = self.cumulative_sizes[-1]
            
    def __len__(self):
        return self.total
    def __getitem__(self, idx: int):
        subject_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')

        if subject_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[subject_idx - 1]
        
        x = torch.from_numpy(self.X_list[subject_idx][local_idx]).float()
        y = torch.tensor(self.y_list[subject_idx][local_idx]).long()
        return x, y