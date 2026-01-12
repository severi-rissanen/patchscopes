#!/usr/bin/env python3
"""
Run Fig. 2 experiments: Next-token prediction across layers.
Compares Logit Lens vs Token Identity Patchscope.
"""
import argparse
import sys
import os

# Add parent directory to path so we can import patchscopes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt

from patchscopes.model import load_model
from patchscopes.data import load_wikitext2, sample_positions
from patchscopes.experiments.next_token import run_next_token_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fig. 2 Experiments - Next-token prediction across layers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Model name or path',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=200,
        help='Number of samples to use from WikiText-2',
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='0,4,8,12,16,20,24,28,31',
        help='Comma-separated layer indices to test',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='fig2_results.png',
        help='Output path for plot',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    return parser.parse_args()


def plot_results(results, output_path):
    """
    Create Fig. 2 style plots.

    Two-panel plot:
    - Top: Precision@1 vs Layer
    - Bottom: Surprisal vs Layer
    Both show Logit Lens and Patchscope curves
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    layers = results['layers']

    # Top: Precision@1
    ax1.plot(layers, results['logit_lens']['precision_at_1'],
             marker='o', label='Logit Lens', linewidth=2, markersize=6)
    ax1.plot(layers, results['patchscope']['precision_at_1'],
             marker='s', label='Token Identity Patchscope', linewidth=2, markersize=6)
    ax1.set_ylabel('Precision@1', fontsize=12)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_title('Next-Token Prediction Accuracy Across Layers', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Bottom: Surprisal
    ax2.plot(layers, results['logit_lens']['surprisal'],
             marker='o', label='Logit Lens', linewidth=2, markersize=6)
    ax2.plot(layers, results['patchscope']['surprisal'],
             marker='s', label='Token Identity Patchscope', linewidth=2, markersize=6)
    ax2.set_ylabel('Surprisal', fontsize=12)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_title('Next-Token Prediction Surprisal Across Layers', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")


def main():
    args = parse_args()

    print("=" * 80)
    print("Fig. 2 Experiments: Next-Token Prediction Across Layers")
    print("=" * 80)
    print()

    # Set random seed
    torch.manual_seed(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Configuration
    dtype = torch.float16 if args.device == 'cuda' else torch.float32

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {dtype}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Random seed: {args.seed}")
    print()

    # Load model
    print(f"Loading model...")
    model = load_model(args.model, device=args.device, dtype=dtype)
    print(f"Model loaded! ({model.cfg.n_layers} layers)")
    print()

    # Load data
    print("Loading WikiText-2 dataset...")
    texts = load_wikitext2(split='test', num_samples=args.num_samples)
    print(f"Loaded {len(texts)} text samples")

    print("Sampling positions for evaluation...")
    samples = sample_positions(model, texts, positions_per_text=1)
    print(f"Using {len(samples)} evaluation samples")
    print()

    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(',')]
    print(f"Testing layers: {layers}")
    print()

    # Run experiment
    print("=" * 80)
    print("Running Experiments")
    print("=" * 80)
    print()

    results = run_next_token_experiment(model, samples, layers)

    # Print summary
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print(f"Layers: {results['layers']}")
    print()
    print("Logit Lens:")
    print(f"  Precision@1: {[f'{x:.3f}' for x in results['logit_lens']['precision_at_1']]}")
    print(f"  Surprisal:   {[f'{x:.3f}' for x in results['logit_lens']['surprisal']]}")
    print()
    print("Token Identity Patchscope:")
    print(f"  Precision@1: {[f'{x:.3f}' for x in results['patchscope']['precision_at_1']]}")
    print(f"  Surprisal:   {[f'{x:.3f}' for x in results['patchscope']['surprisal']]}")
    print()

    # Plot
    plot_results(results, args.output)

    print()
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
