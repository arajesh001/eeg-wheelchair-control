# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from model import EEG_CNN, ndarray_to_tensor, labels_to_tensor, get_device, SpectrogramCNN
from dataset import get_loso_split, EEGDataset

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
    Runs one full pass over the dataloader w/o gradient tracking, returning accuracy. 

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


def evaluate_loss(model: EEG_CNN, dataloader: DataLoader, 
             device: torch.device) -> float:
    """
    Runs one full pass over the dataloader w/o gradient tracking, returning loss. 

    Args:
        model: EEG_CNN 
        dataloader: batched test data (X, y pairs)
        device: cpu or cuda

    Returns:
        Loss as a decimal (0.0-1.0)
    """

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # choose appropriate device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # get preds 
            preds = model(X_batch)

            # compute loss
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item()

    # return avg loss
    return total_loss / len(dataloader)


def run_loso_optimized(subject_ids, input_type:str, n_epochs=50, batch_size=32, data_dir="processed_per_subject"):
    '''
    Memory-efficient version of run_loso -> uses EEGDataset with mmap loading instead of full arrays.
    '''    
    
    device = get_device()
    accuracies = []

    for test_subj in subject_ids:
        train_subjects = [s for s in subject_ids if s != test_subj]

        train_dataset = EEGDataset(train_subjects, data_dir)
        test_dataset = EEGDataset([test_subj], data_dir)

        # splitting for validation loss
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_split, val_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if input_type == "spec":
            model = SpectrogramCNN().to(device)
        elif input_type == "time":
            model = EEG_CNN().to(device)
        else:
            print("Invalid input_type")
            return
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

        # adding early stopping
        best_loss = float('inf')
        counter = 0
        patience = 15
        tol = 0.001
        warmup_epochs = 15
        weights = Path(f"best_model_subject_{test_subj}.pt")

        # delete if its there from previous runs
        if weights.exists():
            weights.unlink()
    

        for epoch in range(n_epochs):
            avg_loss = train(model, train_loader, optimizer, loss_fn, device)
            val_loss = evaluate_loss(model, val_loader, device)
            print(f"[Subject {test_subj}] Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss - tol:
                best_loss = val_loss
                # save the best weights into a temp file ONLY if enough learning has occured
                if epoch + 1 > warmup_epochs:
                    torch.save(model.state_dict(), weights)
                counter = 0

            else:
                counter += 1
            if counter >= patience:
                print(f"[Subject {test_subj}] Early stopping at epoch {epoch+1}")
                break
        
        
        # load in the best weights rather than using most recent
        if weights.exists():
            state_dict = torch.load(weights, weights_only=True)
            model.load_state_dict(state_dict)
        # just use most recent in case the file DNE
        else:
            pass

        acc = evaluate(model, test_loader, device)
        print(f"[Subject {test_subj}] Accuracy: {acc:.4f}\n")
        accuracies.append(acc)

    mean_acc = sum(accuracies) / len(accuracies)
    print(f"Mean LOSO Accuracy: {mean_acc:.4f}")
    return accuracies, mean_acc