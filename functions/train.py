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
    
    """
    Runs one full pass over the dataloader w/o gradient tracking.

    Args:
        model: EEG_CNN 
        dataloader: batched test data (X, y pairs)
        device: cpu or cuda

    Returns:
        Classification accuracy as a decimal (0.0-1.0)
    """

    model.eval()
    correct = 0
    total = 0

    # DONT make computational graph here -> save memory
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # choose appropriate device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # get preds 
            preds = model(X_batch)

            # highest score / sample idx returned -> which maps to class (0-3)
            idx = torch.argmax(preds, dim=1)

            # add to correct and total counts
            correct += (idx == y_batch).sum().item()
            total += len(y_batch)

    # return accuracy as a decimal
    return correct / total

def run_loso(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
             n_epochs: int = 50, batch_size: int = 32) -> tuple[list, float]:
    
    '''
    Evaluates model generalization across subjects using LOSO
    cross validation. Each fold trains on 8 subjects and tests on the held-out
    subject, repeating for all unique subjects, returning per-subject accuracies
    and their mean.

    Args:
        X: [total_epochs, 22, n_freqs, n_times] power arr
        y: [total_epochs] class labels (0-3)
        subjects: [total_epochs] subject ID per epoch
        n_epochs: # of training epochs / fold
        batch_size: # of samples / batch

    Returns:
        accuracies: list of per-subject accuracies as decimals
        mean: mean accuracy across all folds
    '''

    device = get_device()
    accuracies = []

    # EDIT: this loop structure prevents div0 error later -> found using test.ipynb
    for subject_id in np.unique(subjects):
        # split
        X_train, X_test, y_train, y_test = get_loso_split(X, y, subjects, subject_id)

        # dataloader
        train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True)
        test_loader  = make_dataloader(X_test,  y_test,  batch_size, shuffle=False)

        # fresh model + optimizer each fold
        model = EEG_CNN().to(device)

        # Adam optimizer -> best
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        # train n_epochs
        for epoch in range(1, n_epochs + 1):
            loss = train(model, train_loader, optimizer, loss_fn, device)
            print(epoch, loss)
        
        # eval
        acc = evaluate(model, test_loader, device)
        print(subject_id, acc)
        accuracies.append(acc)
    
    total = 0
    mean = np.mean(accuracies)
    return accuracies, mean