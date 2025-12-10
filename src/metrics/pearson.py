import numpy as np

def vectorized_correlation(predictions: np.ndarray, traces: np.ndarray) -> np.ndarray:
    """
    Computes Pearson Correlation Coefficient between predictions and traces.
    
    Args:
        predictions: (n_traces,) array of predicted values (e.g., Hamming Weight).
        traces: (n_traces, n_samples) array of power traces.
        
    Returns:
        correlations: (n_samples,) array of correlation coefficients.
    """
    n_traces = traces.shape[0]
    
    # Center the data (subtract mean)
    # predictions: (n_traces,)
    # traces: (n_traces, n_samples)
    
    mean_pred = np.mean(predictions)
    mean_traces = np.mean(traces, axis=0)
    
    pred_centered = predictions - mean_pred
    traces_centered = traces - mean_traces
    
    # Numerator: sum( (x - mean_x) * (y - mean_y) )
    # We can use matrix multiplication or broadcasting
    # pred_centered is (n_traces,) -> reshape to (n_traces, 1)
    # traces_centered is (n_traces, n_samples)
    # sum over axis 0
    
    numerator = np.sum(pred_centered[:, np.newaxis] * traces_centered, axis=0)
    
    # Denominator: sqrt( sum((x-mean_x)^2) * sum((y-mean_y)^2) )
    ss_pred = np.sum(pred_centered ** 2)
    ss_traces = np.sum(traces_centered ** 2, axis=0)
    
    denominator = np.sqrt(ss_pred * ss_traces)
    
    # Avoid division by zero
    # If denominator is 0, correlation is 0 (or undefined, but 0 is safe for plotting)
    mask = denominator == 0
    denominator[mask] = 1.0
    
    correlations = numerator / denominator
    correlations[mask] = 0.0
    
    return correlations
