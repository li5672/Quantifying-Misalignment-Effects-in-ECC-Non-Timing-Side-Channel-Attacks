import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch
from tqdm import tqdm
from ecc_sca_project.src.generation.generator import ECCTraceGenerator
from ecc_sca_project.src.attacks.spa import SPAAttack
from ecc_sca_project.src.attacks.cpa import CPAAttack
from ecc_sca_project.src.attacks.deep_learning import DeepLearningAttack
from ecc_sca_project.src.utils.visualization import plot_success_curve

def run_all_methods_comparison(jitter=20, n_traces_max=300, n_steps=10, noise=3.0):
    print(f"Running Final Comparison: SPA vs CPA vs CNN")
    print(f"Fixed Jitter: {jitter}")
    print(f"Max Traces: {n_traces_max}, Noise: {noise}")
    
    # 1. Train CNN first (if needed)
    print("Training CNN...")
    n_train = 5000
    n_val = 1000
    trace_len = 1000
    
    # Train on aligned data (J=0) with moderate noise
    gen_train = ECCTraceGenerator(
        trace_length=trace_len,
        n_traces=n_train + n_val,
        tailroom=100,
        max_jitter=0,
        noise_params={'level': 1.5, 'color_factor': 0.5, 'baseline_drift': 0.0}
    )
    
    n_rounds = 10
    keys_train = np.random.randint(0, 2, size=(n_train + n_val, n_rounds))
    traces_train_full, starts_train = gen_train.generate_batch(keys_train)
    
    # Segment data for CNN
    X_train = []
    y_train = []
    pulse_len = 80
    gap = 40
    window_len = 200
    
    for i in range(n_train + n_val):
        for r in range(n_rounds):
            # Add random shift to training data for robustness
            shift = np.random.randint(-5, 6)
            s_add = starts_train[i, r] + pulse_len + gap + shift
            
            if s_add + window_len < traces_train_full.shape[1]:
                window = traces_train_full[i, s_add : s_add + window_len]
                X_train.append(window)
                y_train.append(keys_train[i, r])
                
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    split = int(len(X_train) * 0.8)
    X_t, X_v = X_train[:split], X_train[split:]
    y_t, y_v = y_train[:split], y_train[split:]
    
    cnn = DeepLearningAttack(input_length=window_len)
    cnn.train(X_t, y_t, epochs=5, batch_size=64)
    
    # 2. Benchmark Loop
    n_list = np.linspace(10, n_traces_max, n_steps, dtype=int)
    
    results_spa = []
    results_cpa = []
    results_cnn = []
    
    n_trials = 20
    
    print(f"\nEvaluating all methods at J={jitter}...")
    
    for n in tqdm(n_list):
        success_spa = 0
        success_cpa = 0
        success_cnn = 0
        
        # Generate data
        gen = ECCTraceGenerator(
            trace_length=trace_len,
            n_traces=n * n_trials,
            tailroom=100,
            max_jitter=jitter,
            noise_params={'level': noise, 'color_factor': 0.5, 'baseline_drift': 0.05}
        )
        
        keys = np.random.randint(0, 2, size=(n_trials, n_rounds))
        bits_repeated = np.repeat(keys, n, axis=0)
        
        traces_all, starts_all = gen.generate_batch(bits_repeated, seed=100+jitter+n)
        
        # Initialize Attacks
        spa = SPAAttack(pulse_len=80, gap=40)
        cpa = CPAAttack(pulse_len=80, gap=40)
        
        for t in range(n_trials):
            batch_traces = traces_all[t*n : (t+1)*n]
            batch_starts = starts_all[t*n : (t+1)*n]
            
            # --- SPA Attack ---
            # Average traces
            avg_trace = np.mean(batch_traces, axis=0)
            # Use nominal starts (blind)
            round_len = pulse_len * 2 + gap
            nominal_starts = np.arange(n_rounds) * round_len + 10
            
            pred_spa, _ = spa.recover_key(avg_trace, nominal_starts)
            if np.array_equal(pred_spa, keys[t]):
                success_spa += 1
                
            # --- CPA Attack ---
            # Use noisy starts to simulate misalignment
            if jitter > 0:
                jitter_error = np.random.randint(-jitter, jitter+1, size=batch_starts.shape)
                noisy_starts = batch_starts + jitter_error
            else:
                noisy_starts = batch_starts
                
            pred_cpa_all, _ = cpa.recover_key(batch_traces, noisy_starts)
            votes_cpa = np.sum(pred_cpa_all, axis=0)
            final_pred_cpa = (votes_cpa > (n / 2)).astype(int)
            
            if np.array_equal(final_pred_cpa, keys[t]):
                success_cpa += 1
                
            # --- CNN Attack ---
            # Segment windows using noisy starts (fair comparison with CPA)
            X_batch = []
            for i in range(n):
                for r in range(n_rounds):
                    s = noisy_starts[i, r] + pulse_len + gap
                    if s + window_len < batch_traces.shape[1]:
                        window = batch_traces[i, s : s + window_len]
                        X_batch.append(window)
                    else:
                        pad = np.zeros(window_len)
                        rem = batch_traces.shape[1] - s
                        if rem > 0:
                            pad[:rem] = batch_traces[i, s:]
                        X_batch.append(pad)
                        
            X_batch = np.array(X_batch)
            pred_cnn_raw = cnn.recover_key(X_batch)
            pred_cnn_reshaped = pred_cnn_raw.reshape(n, n_rounds)
            votes_cnn = np.sum(pred_cnn_reshaped, axis=0)
            final_pred_cnn = (votes_cnn > (n / 2)).astype(int)
            
            if np.array_equal(final_pred_cnn, keys[t]):
                success_cnn += 1
                
        results_spa.append(success_spa / n_trials)
        results_cpa.append(success_cpa / n_trials)
        results_cnn.append(success_cnn / n_trials)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_list, results_spa, 'o-', label='SPA (Otsu)', color='#e41a1c', linewidth=2)
    ax.plot(n_list, results_cpa, 's-', label='CPA (Matched Filter)', color='#377eb8', linewidth=2)
    ax.plot(n_list, results_cnn, '^-', label='CNN (Deep Learning)', color='#4daf4a', linewidth=2)
    
    ax.set_xlabel("Number of Traces")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Method Comparison: SPA vs CPA vs CNN (Jitter={jitter}, Noise={noise})")
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join("results", "final_comparison_all_methods.png")
    fig.savefig(output_path)
    print(f"Saved comparison plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter", type=int, default=20, help="Fixed jitter level")
    parser.add_argument("--noise", type=float, default=3.0, help="Noise level")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    run_all_methods_comparison(jitter=args.jitter, noise=args.noise)
