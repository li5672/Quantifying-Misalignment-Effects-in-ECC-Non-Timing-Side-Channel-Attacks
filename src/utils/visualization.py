import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple

def plot_success_curve(
    n_list: List[int], 
    success_rates: List[float], 
    confidence_intervals: Optional[List[Tuple[float, float]]] = None,
    label: str = "Attack",
    color: str = "blue",
    ax=None,
    add_threshold_line: bool = True,
    linestyle: str = '-',
    marker: str = 'o'
):
    """
    Plots S(n) curve with optional confidence intervals.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(n_list, success_rates, marker=marker, linestyle=linestyle, label=label, color=color)
    
    if confidence_intervals:
        lower = [ci[0] for ci in confidence_intervals]
        upper = [ci[1] for ci in confidence_intervals]
        ax.fill_between(n_list, lower, upper, color=color, alpha=0.2)
        
    if add_threshold_line:
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% Success')
    ax.set_xlabel("Number of Traces (N)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_jitter_degradation(jitter_list: List[int], r_j_values: List[float], ax=None):
    """
    Plots r(j) vs Jitter.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(jitter_list, r_j_values, marker='s', color='red', label='Degradation Factor r(j)')
    ax.set_xlabel("Jitter (samples)")
    ax.set_ylabel("r(j) = N80(j) / N80(0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax
