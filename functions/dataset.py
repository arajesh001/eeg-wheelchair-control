# =============================================================================
# IMPORTS
# =============================================================================

import mne
import numpy as np

# =============================================================================
# FUNCTIONS
# =============================================================================

def log_normalize(power: np.ndarray) -> np.ndarray:
    pass

def load_subject(subject_id: int, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    pass

def build_dataset(data_dir: str, subject_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pass

def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> tuple:
    pass