import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from ecc_sca_project.src.generation.generator import ECCTraceGenerator
from ecc_sca_project.src.attacks.cpa import CPAAttack
from ecc_sca_project.src.metrics.statistics import wilson_score_interval
from ecc_sca_project.src.utils.visualization import plot_success_curve

def run_cpa_benchmark(max_jitter_list=[0, 10, 20, 30], n_traces_max=300, n_steps=10, noise=3.0):
    print(f"Running CPA Benchmark...")
    print(f"Jitter Levels: {max_jitter_list}")
    print(f"Max Traces: {n_traces_max}, Noise: {noise}")
    
    n_list = np.linspace(10, n_traces_max, n_steps, dtype=int)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    for i, jitter in enumerate(max_jitter_list):
        print(f"\nEvaluating Jitter = {jitter}...")
        success_rates = []
        
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        n_trials = 20 # Number of trials to estimate success rate
        
        for n in tqdm(n_list, desc=f"J={jitter}"):
            successes = 0
            
            # Generate data for n_trials
            # Total traces: n * n_trials
            gen = ECCTraceGenerator(
                trace_length=1000,
                n_traces=n * n_trials,
                tailroom=100,
                max_jitter=jitter,
                noise_params={'level': noise, 'color_factor': 0.5, 'baseline_drift': 0.05}
            )
            
            n_rounds = 10
            keys = np.random.randint(0, 2, size=(n_trials, n_rounds))
            bits_repeated = np.repeat(keys, n, axis=0)
            
            traces_all, starts_all = gen.generate_batch(bits_repeated, seed=42+jitter+n)
            
            cpa = CPAAttack(pulse_len=80, gap=40)
            
            for t in range(n_trials):
                batch_traces = traces_all[t*n : (t+1)*n]
                batch_starts = starts_all[t*n : (t+1)*n]
                
                # Run CPA
                # We use the batch_starts (ground truth) but CPA is affected by jitter 
                # because the signal within the window is shifted.
                # Note: If we pass exact starts, we are effectively aligning the window start.
                # But if jitter exists, the "Add" pulse might be shifted relative to "Double" pulse?
                # In our generator, `start` is the beginning of Double.
                # `Add` is at `start + pulse + gap`.
                # If jitter is per-operation, the distance between Double and Add might vary?
                # Let's check generator logic briefly in mind:
                # `current_idx` updates.
                # If generator adds jitter *between* pulses, then CPA degrades.
                # If generator only adds jitter to `current_idx` *before* the round, 
                # then `starts` captures it, and CPA with `starts` would be perfect.
                # However, usually "Jitter" implies we don't know the exact start.
                # But here we pass `starts`.
                # If we want to simulate "Misalignment", we should probably ADD noise to `starts`?
                # Or assume `starts` is just the "Trigger" and the actual signal is jittered relative to it.
                # In our generator, `starts` IS the actual signal start.
                # So passing `starts` is "Aligned Attack".
                # To simulate "Misaligned Attack", we should pass `starts + random_jitter`.
                
                # Let's add simulated misalignment to the starts passed to CPA.
                # The attacker thinks the start is at `s`, but it's actually at `s + j`.
                # So we pass `s` (nominal) but the trace has signal at `s+j`.
                # Wait, `starts_all` contains the ACTUAL positions.
                # So we should pass `starts_all + error`?
                # Or `starts_all - error`?
                # If we pass `starts_all`, we are cheating (Perfect Alignment).
                # We should pass "Nominal Starts".
                # But since length varies, nominal is hard.
                # Let's simply perturb the `starts_all` by random jitter before passing to CPA.
                # This simulates the attacker guessing the start with some error.
                
                if jitter > 0:
                    # Add uniform jitter error [-J, J]
                    jitter_error = np.random.randint(-jitter, jitter+1, size=batch_starts.shape)
                    noisy_starts = batch_starts + jitter_error
                    # Clip to valid range? CPA handles bounds.
                else:
                    noisy_starts = batch_starts
                
                # Run CPA with noisy starts
                predicted_bits_all, _ = cpa.recover_key(batch_traces, noisy_starts)
                
                # Vote
                votes = np.sum(predicted_bits_all, axis=0)
                final_pred = (votes > (n / 2)).astype(int)
                
                if np.array_equal(final_pred, keys[t]):
                    successes += 1
            
            success_rates.append(successes / n_trials)
            
        plot_success_curve(n_list, success_rates, None, label=f"J={jitter}", color=color, linestyle=linestyle, ax=ax, add_threshold_line=(i==0))
        
    ax.set_title(f"CPA Success Rate vs Traces (Noise={noise})")
    output_path = os.path.join("results", "cpa_success_curves.png")
    fig.savefig(output_path)
    print(f"Saved results/cpa_success_curves.png")

if __name__ == "__main__":
    # Ensure results dir exists
    os.makedirs("results", exist_ok=True)
    run_cpa_benchmark()
