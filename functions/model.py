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


        # BLOCK 3 -> separable conv (depthwise + pointwise)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=16 
        )
        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 1)
        )
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        # FLATTENING + LINEAR LAYER
        # pass a dummy tensor through the conv layers to get the right size
        with torch.no_grad():
            dummy = torch.zeros(1, 22, 65, 626)
            dummy = self.pool(F.elu(self.bn2(self.conv2(F.elu(self.bn1(self.conv1(dummy)))))))
            dummy = self.pool2(F.elu(self.bn3(self.conv4(self.conv3(dummy)))))
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

        # BLOCK 3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
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
    
###### 


class SpectrogramCNN(nn.Module):
    def __init__(self, n_channels=22, n_freqs=65, n_times=626, n_classes=4):
        super().__init__()

        # BLOCK 0 - freq compression
        # kernel (8,1) -> only moves along frequency axis; ignores time
        # groups=n_channels -> each electrode processed independently, no mixing yet
        # depthwise conv -> shared weights per channel
        self.freq_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * 2,
            kernel_size=(8, 1),
            padding=(4, 0),
            groups=n_channels
        )
        self.freq_bn = nn.BatchNorm2d(n_channels * 2)
        # compress 65 freq bins -> ~16
        self.freq_pool = nn.AvgPool2d(kernel_size=(4, 1))

        # BLOCK 1 - temporal conv
        # kernel (1,25) -> only moves along time axis
        # 25 samples at 250hz = 100ms window
        self.temp_conv = nn.Conv2d(
            in_channels=n_channels * 2,
            out_channels=40,
            kernel_size=(1, 25),
            padding=(0, 12)
        )
        self.temp_bn = nn.BatchNorm2d(40)

        # BLOCK 2 - spatial conv
        # kernel (16,1) -> collapses across frequency dimension (16 bins after pool)
        self.spat_conv = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=(16, 1)
        )
        self.spat_bn = nn.BatchNorm2d(40)

        # pool across time -> 75 sample window, stride 15
        # 75 samples at 250hz = 300ms, stride 15 = 60ms steps
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=0.5)

        # compute flat size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_freqs, n_times)
            dummy = self._forward_features(dummy)
            flat_size = dummy.flatten(start_dim=1).shape[1]

        # go straight to classifier, no hidden layer -> scnn style
        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        # block 0
        x = self.freq_conv(x)
        x = self.freq_bn(x)
        x = F.elu(x)
        x = self.freq_pool(x)

        # block 1
        x = self.temp_conv(x)
        x = self.temp_bn(x)
        x = F.elu(x)

        # block 2
        x = self.spat_conv(x)
        x = self.spat_bn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SpectrogramCNN()
    dummy = torch.randn(8, 22, 65, 626)
    out = model(dummy)
    print(out.shape)  # should be [8, 4]
