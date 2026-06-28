# =============================================================================
# IMPORTS
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# =============================================================================
# CLASSES
# =============================================================================

class EEG_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # BLOCK 1 -> temporal filter across time 
        self.conv1 = nn.Conv2d(
            in_channels = 22,
            out_channels = 8,
            kernel_size = (1, 64),
            padding = (0, 32)
        )

        self.bn1 = nn.BatchNorm2d(num_features=8)

        # BLOCK 2 -> spatial filter across freqs 
        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = (22, 1),
            padding = 0
        )

        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.pool = nn.AvgPool2d(
            kernel_size = 2,
            stride = 2
        )

        # FLATTENING + LINEAR LAYER
        # pass a dummy tensor through the conv layers to get the right size
        with torch.no_grad():
            dummy = torch.zeros(1, 22, 65, 626)
            dummy = self.pool(F.elu(self.bn2(self.conv2(
                F.elu(self.bn1(self.conv1(dummy)))))))
            flat_size = dummy.flatten(start_dim=1).shape[1]

        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 4)

        # prevent overfitting w/random deactivation
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_conv = nn.Dropout2d(p=0.25)


    def forward(self, x):
        # BLOCK 1 
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        # BLOCK 2 
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        # FLATTENING + LINEAR LAYER 1
        # [batch, flat_size] —> computed dynamically in __init__
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        # [8, 128]
        x = self.fc1(x)
        x = F.elu(x)

        # LINEAR LAYER 2 
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# DUMMY CODE TO TEST IF CNN WORKS
if __name__ == "__main__":
    model = EEG_CNN()
    dummy = torch.randn(8, 22, 65, 626)
    out = model(dummy)
    print(out.shape)  
    # ^^ should be [8, 4]


# =============================================================================
# FUNCTIONS
# =============================================================================

def ndarray_to_tensor(X: np.ndarray) -> torch.Tensor:
    # float32 -> required by Conv2d
    return torch.from_numpy(X).float()

def labels_to_tensor(y: np.ndarray) -> torch.Tensor:
    # int64 -> required by CrossEntropyLoss (see train.py)
    return torch.from_numpy(y).long()

def get_device() -> torch.device:
    # UPDATED FOR MORE OPTIONS
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
