import pytest
import numpy as np
from ecc_sca_project.src.utils.alignment import CrossCorrelationAligner

def test_shift_logic():
    """
    Verify the internal _shift_trace method.
    """
    aligner = CrossCorrelationAligner()
    trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Shift right by 1
    shifted = aligner._shift_trace(trace, 1)
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(shifted, expected)
    
    # Shift left by 1
    shifted = aligner._shift_trace(trace, -1)
    expected = np.array([2.0, 3.0, 4.0, 5.0, 0.0])
    assert np.allclose(shifted, expected)

def test_alignment_simple_pulse():
    """
    Verify alignment of a simple pulse.
    """
    aligner = CrossCorrelationAligner(max_shift=10)
    length = 100
    
    # Reference: Pulse at 50
    ref = np.zeros(length)
    ref[50] = 1.0
    
    # Trace: Pulse at 55 (Shifted right by 5)
    trace = np.zeros(length)
    trace[55] = 1.0
    
    traces = np.vstack([ref, trace])
    
    aligned = aligner.align(traces, reference_idx=0)
    
    # Trace 1 should now have pulse at 50
    assert aligned[1, 50] > 0.9
    assert aligned[1, 55] < 0.1

def test_alignment_random_jitter():
    """
    Verify alignment reduces variance on a batch of jittery traces.
    """
    n_traces = 10
    length = 100
    aligner = CrossCorrelationAligner(max_shift=20)
    
    # Create base trace
    base = np.exp(-0.5 * ((np.arange(length) - length//2) / 5) ** 2)
    
    traces = []
    for i in range(n_traces):
        shift = np.random.randint(-10, 11)
        # Create shifted trace
        tr = np.zeros(length)
        if shift >= 0:
            tr[shift:] = base[:length-shift]
        else:
            s = -shift
            tr[:length-s] = base[s:]
        traces.append(tr)
        
    traces = np.array(traces)
    
    # Variance before alignment
    var_before = np.var(traces, axis=0).sum()
    
    aligned = aligner.align(traces, reference_idx=0)
    
    # Variance after alignment
    var_after = np.var(aligned, axis=0).sum()
    
    # Alignment should reduce variance (sharpen the mean trace)
    assert var_after < var_before, f"Alignment failed to reduce variance. Before: {var_before}, After: {var_after}"
