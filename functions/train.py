# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import EEG_CNN, ndarray_to_tensor, labels_to_tensor, get_device
from dataset import get_loso_split

# =============================================================================
# LOSS FUNCTION
# =============================================================================


# =============================================================================
# FUNCTIONS
# =============================================================================


def make_dataloader(X: np.ndarray, y: np.ndarray, 
                    batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Takens in X and y ndarrays and converts to torch tensors, and passes them
    into TensorDataset and DataLoader. 

    Args:
    X: np.ndarray [total_epochs, 22, n_freqs, n_times]
    y: np.ndarray [total_epochs] integer class labels
    batch_size: 32 by default 
    shuffle: true on default

    Returns: DataLoader object
    """
    
    X = ndarray_to_tensor(X)
    y = labels_to_tensor(y)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return loader

def train(model: EEG_CNN, dataloader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          loss_fn: nn.Module, device: torch.device) -> float:
    pass

def evaluate(model: EEG_CNN, dataloader: DataLoader, 
             device: torch.device) -> float:
    pass

def run_loso(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
             n_epochs: int = 50, batch_size: int = 32) -> tuple[list, float]:
    pass