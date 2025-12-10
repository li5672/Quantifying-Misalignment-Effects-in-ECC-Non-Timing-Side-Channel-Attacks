import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from ecc_sca_project.src.generation.generator import ECCTraceGenerator
from ecc_sca_project.src.attacks.deep_learning import DeepLearningAttack

def run_demo(n_train=500, n_test=100, jitter=30, noise=0.5, epochs=10):
    print(f"Running CNN Demo with Train={n_train}, Test={n_test}, Jitter={jitter}, Noise={noise}...")
    
    # 1. Generate Training Data (with Jitter)
    print("Generating training data...")
    gen_train = ECCTraceGenerator(
        trace_length=1000, 
        n_traces=n_train,
        tailroom=100,
        max_jitter=jitter,
        noise_params={'level': noise, 'color_factor': 0.5, 'baseline_drift': 0.05}
    )
    
    # We need labels (bits). Let's train on the first round bit?
    # Or train on random bits and predict random bits.
    # The CNN takes a trace segment and predicts the bit.
    # We need to slice the traces into examples.
    # For simplicity in this demo, let's generate traces with 1 round only.
    
    n_rounds = 1
    bits_train = np.random.randint(0, 2, size=(n_train, n_rounds))
    traces_train, _ = gen_train.generate_batch(bits_train, seed=42)
    
    # Labels are just the bits
    y_train = bits_train.flatten()
    
    # 2. Generate Test Data (with Jitter)
    print("Generating test data...")
    gen_test = ECCTraceGenerator(
        trace_length=1000, 
        n_traces=n_test,
        tailroom=100,
        max_jitter=jitter,
        noise_params={'level': noise, 'color_factor': 0.5, 'baseline_drift': 0.05}
    )
    bits_test = np.random.randint(0, 2, size=(n_test, n_rounds))
    traces_test, _ = gen_test.generate_batch(bits_test, seed=123)
    y_test = bits_test.flatten()
    
    # 3. Train CNN
    print(f"Training CNN for {epochs} epochs...")
    # Input length is the full trace length
    input_len = traces_train.shape[1]
    attacker = DeepLearningAttack(input_length=input_len, device="cpu") # Use CPU for demo compatibility
    
    attacker.train(traces_train, y_train, epochs=epochs, batch_size=32)
    
    # 4. Evaluate
    print("Evaluating...")
    preds = attacker.recover_key(traces_test)
    accuracy = np.mean(preds == y_test)
    print(f"CNN Accuracy (Jitter={jitter}): {accuracy * 100:.2f}%")
    
    # 5. Visualize
    # Plot a few test traces and their predictions
    plt.figure(figsize=(12, 6))
    
    # Sort by class for visualization
    idx_0 = np.where(y_test == 0)[0][:3]
    idx_1 = np.where(y_test == 1)[0][:3]
    
    indices = np.concatenate([idx_0, idx_1])
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i+1)
        plt.plot(traces_test[idx], label=f"True: {y_test[idx]}")
        plt.title(f"Pred: {preds[idx]} (Correct: {preds[idx]==y_test[idx]})")
        plt.legend()
        
    plt.tight_layout()
    output_path = os.path.join("results", f"cnn_demo_j{jitter}.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter", type=int, default=30, help="Max jitter amount")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise level")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()
    
    run_demo(jitter=args.jitter, noise=args.noise, epochs=args.epochs)
