import pytest
import numpy as np
from ecc_sca_project.src.metrics.statistics import wilson_score_interval
from ecc_sca_project.src.metrics.calculator import MetricCalculator

def test_wilson_score():
    """
    Verify Wilson score interval calculation.
    """
    # 100 trials, 50 successes -> 0.5
    p, lower, upper = wilson_score_interval(50, 100)
    assert p == 0.5
    assert lower < 0.5
    assert upper > 0.5
    # 95% CI for 0.5 with N=100 is approx [0.40, 0.60]
    assert 0.39 < lower < 0.41
    assert 0.59 < upper < 0.61

def test_n80_interpolation():
    """
    Verify N80 interpolation.
    """
    n_list = [10, 20, 30, 40, 50]
    success_rates = [0.5, 0.6, 0.7, 0.9, 1.0]
    
    # 0.8 is between 30 (0.7) and 40 (0.9)
    # Linear interp: 0.8 is exactly halfway between 0.7 and 0.9
    # So N80 should be halfway between 30 and 40 -> 35
    
    n80 = MetricCalculator.compute_n80(n_list, success_rates)
    assert abs(n80 - 35.0) < 0.1

def test_n80_not_reached():
    """
    Verify N80 returns inf if success rate never reaches 0.8.
    """
    n_list = [10, 20]
    success_rates = [0.5, 0.6]
    
    n80 = MetricCalculator.compute_n80(n_list, success_rates)
    assert n80 == float('inf')

def test_jitter_degradation():
    """
    Verify r(j) calculation.
    """
    n80_base = 100
    n80_j = 200
    
    r = MetricCalculator.compute_jitter_degradation(n80_j, n80_base)
    assert r == 2.0
