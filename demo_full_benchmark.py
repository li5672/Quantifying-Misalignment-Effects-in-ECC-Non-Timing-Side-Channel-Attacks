import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from ecc_sca_project.src.generation.generator import ECCTraceGenerator
from ecc_sca_project.src.attacks.spa import SPAAttack
from ecc_sca_project.src.metrics.statistics import wilson_score_interval
from ecc_sca_project.src.metrics.calculator import MetricCalculator
from ecc_sca_project.src.utils.visualization import plot_success_curve, plot_jitter_degradation

def run_benchmark(max_jitter_list=[0, 10, 20, 30], n_traces_max=200, n_steps=10, noise=0.5):
    print(f"Running Full Benchmark...")
    print(f"Jitter Levels: {max_jitter_list}")
    print(f"Max Traces: {n_traces_max}, Noise: {noise}")
    
    # Define N list for S(n) curve
    n_list = np.linspace(10, n_traces_max, n_steps, dtype=int)
    
    n80_results = []
    
    # Setup Plot
    fig_sn, ax_sn = plt.subplots(figsize=(10, 6))
    
    # Define colors for different jitter levels (Rainbow pattern)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    # Define distinct linestyles for black and white readability
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    for i, jitter in enumerate(max_jitter_list):
        print(f"\nEvaluating Jitter = {jitter}...")
        success_rates = []
        confidence_intervals = []
        
        # Select color and linestyle (cycle if needed)
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # For each N, we need to estimate success rate.
        # Success rate is: Probability that the attack recovers the FULL KEY correctly.
        # Or Bit Success Rate?
        # Usually Key Success Rate.
        # To estimate S(n), we need multiple TRIALS of the attack with n traces.
        n_trials = 50 # Number of experiments to estimate probability
        
        for n in tqdm(n_list, desc=f"J={jitter}"):
            successes = 0
            
            # We need to generate data for n_trials
            # Each trial uses n traces.
            # Total traces needed: n * n_trials
            # To save time, we can generate a large pool and sample?
            # Or just generate on fly.
            
            # Let's generate on fly to be safe.
            gen = ECCTraceGenerator(
                trace_length=1000,
                n_traces=n * n_trials,
                tailroom=100,
                max_jitter=jitter,
                noise_params={'level': noise, 'color_factor': 0.5, 'baseline_drift': 0.05}
            )
            
            # Generate random keys for each trial?
            # Or same key?
            # Usually we attack a fixed key with n traces.
            # So for each trial, we pick a key, generate n traces, attack.
            
            # Efficient generation:
            # Generate (n_trials, n, n_rounds) bits?
            # Generator expects (n_total_traces, n_rounds).
            # We can reshape.
            
            n_rounds = 10 # Keep it short for speed
            
            # Correct Generation Logic
            # We need n_trials. For each, 1 key, n traces.
            # Total traces: n * n_trials.
            # Key 1 -> n traces
            # Key 2 -> n traces
            # ...
            
            # Generate keys
            keys = np.random.randint(0, 2, size=(n_trials, n_rounds))
            # Repeat keys: (n_trials, n, n_rounds) -> flatten
            bits_repeated = np.repeat(keys, n, axis=0) # (n_trials*n, n_rounds)
            
            traces_all, starts_all = gen.generate_batch(bits_repeated, seed=42+jitter+n)
            
            # Run Attack (SPA with Averaging)
            spa = SPAAttack(pulse_len=80, gap=40)
            
            for t in range(n_trials):
                batch_traces = traces_all[t*n : (t+1)*n]
                # Average traces
                avg_trace = np.mean(batch_traces, axis=0)
                
                # Let's use nominal starts to simulate "blind" attack
                pulse_len = 80; gap = 40
                round_len = pulse_len * 2 + gap
                nominal_starts = np.arange(n_rounds) * round_len + 10
                
                predicted_bits, _ = spa.recover_key(avg_trace, nominal_starts)
                
                actual_key = keys[t]
                if np.array_equal(predicted_bits, actual_key):
                    successes += 1
            
            # Calculate stats
            p, lower, upper = wilson_score_interval(successes, n_trials)
            success_rates.append(p)
            confidence_intervals.append((lower, upper))
            
        # Plot curve
        # User requested to remove confidence intervals (min/max shading)
        plot_success_curve(n_list, success_rates, None, label=f"J={jitter}", color=color, linestyle=linestyle, ax=ax_sn, add_threshold_line=(i==0))
        
        # Compute N80
        n80 = MetricCalculator.compute_n80(n_list, success_rates)
        n80_results.append(n80)
        print(f"  N80(J={jitter}) = {n80:.1f}")

    ax_sn.set_title(f"Success Rate S(n) vs Traces (Noise={noise})")
    fig_sn.savefig(os.path.join("results", "benchmark_success_curves.png"))
    print("Saved results/benchmark_success_curves.png")
    
    # Compute r(j)
    print("\nComputing Jitter Degradation r(j)...")
    n80_baseline = n80_results[0] # J=0
    r_j_values = []
    for n80 in n80_results:
        r = MetricCalculator.compute_jitter_degradation(n80, n80_baseline)
        r_j_values.append(r)
        
    # Plot r(j)
    fig_rj, ax_rj = plt.subplots(figsize=(8, 5))
    plot_jitter_degradation(max_jitter_list, r_j_values, ax=ax_rj)
    ax_rj.set_title("Jitter Degradation Factor r(j)")
    fig_rj.savefig(os.path.join("results", "benchmark_jitter_degradation.png"))
    print("Saved results/benchmark_jitter_degradation.png")
    
    # Print Summary
    print("\nSummary Table:")
    print(f"{'Jitter':<10} | {'N80':<10} | {'r(j)':<10}")
    print("-" * 36)
    for j, n80, r in zip(max_jitter_list, n80_results, r_j_values):
        print(f"{j:<10} | {n80:<10.1f} | {r:<10.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=1.0, help="Noise level")
    args = parser.parse_args()
    
    run_benchmark(noise=args.noise)
