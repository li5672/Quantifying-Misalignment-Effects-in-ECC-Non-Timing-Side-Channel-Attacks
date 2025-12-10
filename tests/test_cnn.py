import pytest
import numpy as np
import torch
from ecc_sca_project.src.attacks.cnn import SCACNN
from ecc_sca_project.src.attacks.deep_learning import DeepLearningAttack

def test_cnn_shapes():
    """
    Verify input/output shapes of the CNN.
    """
    batch_size = 8
    length = 1000
    model = SCACNN(input_length=length)
    
    x = torch.randn(batch_size, length)
    output = model(x)
    
    assert output.shape == (batch_size, 1), f"Expected output shape (8, 1), got {output.shape}"

def test_overfit_clean_data():
    """
    Verify model can learn a simple pattern (overfit small batch).
    """
    length = 100
    n_samples = 20
    
    # Create synthetic data: Class 0 is low noise, Class 1 has a pulse
    X = np.random.normal(0, 0.1, size=(n_samples, length))
    y = np.random.randint(0, 2, size=n_samples)
    
    # Add pulse for class 1
    pulse = np.exp(-0.5 * ((np.arange(length) - length//2) / 5) ** 2)
    for i in range(n_samples):
        if y[i] == 1:
            X[i] += pulse
            
    attacker = DeepLearningAttack(input_length=length, device="cpu") # Force CPU for test
    
    # Train for enough epochs to overfit
    attacker.train(X, y, epochs=50, batch_size=4)
    
    # Predict on same data
    preds = attacker.recover_key(X)
    
    accuracy = np.mean(preds == y)
    assert accuracy == 1.0, f"CNN failed to overfit clean data. Accuracy: {accuracy}"
