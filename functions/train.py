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

loss_fn = nn.CrossEntropyLoss()

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
    """
    Runs one full epoch of training over all batches in the dataloader.

    Args:
        model: EEG_CNN instance in training mode
        dataloader: batched training data (X, y pairs)
        optimizer: Adam
        loss_fn: CrossEntropyLoss 
        device: cpu or cuda -> from get_device

    Returns:
        Average loss across all batches for this epoch
    """

    model.train()
    running_loss = 0

    for X_batch, y_batch in dataloader:
        # choose appropriate device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # zero grads initially
        optimizer.zero_grad()

        # get model predictions
        preds = model(X_batch)

        # compute cross entropy loss
        loss = loss_fn(preds, y_batch)

        # find the gradients
        loss.backward()

        # update weights accordingly
        optimizer.step()
        
        # add to running loss
        running_loss += loss.item()
    
    # return avg loss
    return running_loss / len(dataloader)


def evaluate(model: EEG_CNN, dataloader: DataLoader, 
             device: torch.device) -> float:
    pass

def run_loso(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
             n_epochs: int = 50, batch_size: int = 32) -> tuple[list, float]:
    pass