import torch
import torch.nn as nn

class SCACNN(nn.Module):
    """
    1D-CNN for Side-Channel Analysis.
    Designed to be robust against jitter using Average Pooling.
    """
    def __init__(self, input_length: int, num_classes: int = 1):
        super(SCACNN, self).__init__()
        
        # Architecture inspired by Zaid et al. / ASCAD but simplified for this benchmark
        # Key feature: Average Pooling for translation invariance
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        
        # Calculate flattened size
        # Input length L -> L/2 -> L/4 -> L/8
        self._to_linear = 16 * (input_length // 8)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Sigmoid is applied in loss function (BCEWithLogitsLoss) or explicitly if needed
        # We output logits here
        
    def forward(self, x):
        # x shape: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.features(x)
        x = self.classifier(x)
        return x
