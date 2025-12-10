import pytest
import numpy as np
from ecc_sca_project.src.generation.generator import ECCTraceGenerator

def test_trace_length_consistency():
    """
    Verify that generated traces have the expected length.
    """
    n_traces = 10
    n_rounds = 5
    # These params should match what's inside generator (or we should expose them)
    # Currently generator hardcodes pulse_len=80, gap=40.
    # round_len = 200.
    # total = 200 * 5 + 2*jitter + tailroom
    
    tailroom = 100
    max_jitter = 10
    
    gen = ECCTraceGenerator(
        trace_length=1000, # This param is currently ambiguous in my impl, need to fix
        n_traces=n_traces,
        tailroom=tailroom,
        max_jitter=max_jitter,
        noise_params={'level': 0.0}
    )
    
    bits = np.random.randint(0, 2, size=(n_traces, n_rounds))
    traces, starts = gen.generate_batch(bits)
    
    # Expected length calculation
    pulse_len = 80
    gap = 40
    round_len = pulse_len * 2 + gap
    expected_len = round_len * n_rounds + 2 * max_jitter + tailroom
    
    assert traces.shape == (n_traces, expected_len)
    assert starts.shape == (n_traces, n_rounds)

def test_deterministic_no_noise_jitter():
    """
    If J=0 and Noise=0, traces with same bits should be identical.
    """
    n_traces = 5
    n_rounds = 10
    gen = ECCTraceGenerator(
        trace_length=1000,
        n_traces=n_traces,
        tailroom=50,
        max_jitter=0,
        noise_params={'level': 0.0, 'color_factor': 0.0, 'baseline_drift': 0.0}
    )
    
    # Same bits for all traces
    bits_row = np.random.randint(0, 2, size=n_rounds)
    bits = np.tile(bits_row, (n_traces, 1))
    
    traces, _ = gen.generate_batch(bits, seed=42)
    
    # Check all traces are equal to the first one
    for i in range(1, n_traces):
        assert np.allclose(traces[i], traces[0]), f"Trace {i} differs from Trace 0"

def test_double_always_present():
    """
    Verify that the 'Double' pulse is always present.
    """
    n_traces = 1
    n_rounds = 1
    gen = ECCTraceGenerator(
        trace_length=1000,
        n_traces=n_traces,
        tailroom=50,
        max_jitter=0,
        noise_params={'level': 0.0}
    )
    
    bits = np.array([[0]]) # Bit 0, so only Double
    traces, starts = gen.generate_batch(bits)
    
    # Check peak at Double position
    start_idx = starts[0, 0]
    # Pulse center is at start_idx + 40 (pulse_len/2)
    peak_idx = start_idx + 40
    
    assert traces[0, peak_idx] > 0.5, "Double pulse peak missing"

def test_add_pulse_logic():
    """
    Verify 'Add' pulse appears only when bit=1.
    """
    n_traces = 2
    n_rounds = 1
    gen = ECCTraceGenerator(
        trace_length=1000,
        n_traces=n_traces,
        tailroom=50,
        max_jitter=0,
        noise_params={'level': 0.0}
    )
    
    bits = np.array([[0], [1]])
    traces, starts = gen.generate_batch(bits)
    
    # Add pulse is at start + pulse_len + gap + pulse_len/2
    # 80 + 40 + 40 = 160
    add_peak_offset = 80 + 40 + 40
    
    start_0 = starts[0, 0]
    start_1 = starts[1, 0]
    
    # Trace 0 (bit 0): Should be near 0 at Add position
    assert traces[0, start_0 + add_peak_offset] < 0.1
    
    # Trace 1 (bit 1): Should be high at Add position
    assert traces[1, start_1 + add_peak_offset] > 0.5
