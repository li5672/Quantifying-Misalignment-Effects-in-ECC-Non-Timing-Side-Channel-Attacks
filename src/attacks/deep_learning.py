import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple
from .cnn import SCACNN

class DeepLearningAttack:
    """
    Wraps the training and attack logic for the CNN.
    """
    def __init__(self, input_length: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = SCACNN(input_length=input_length).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, traces: np.ndarray, labels: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """
        Trains the model.
        
        Args:
            traces: (n_samples, length)
            labels: (n_samples,) 0 or 1
        """
        self.model.train()
        
        # Convert to Tensor
        X = torch.tensor(traces, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def recover_key(self, traces: np.ndarray) -> np.ndarray:
        """
        Predicts bits for the given traces.
        
        Args:
            traces: (n_samples, length)
            
        Returns:
            predictions: (n_samples,) 0 or 1
        """
        self.model.eval()
        X = torch.tensor(traces, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
        return preds.cpu().numpy().flatten().astype(int)
