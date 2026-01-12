#!/usr/bin/env python3
"""
Run full multi-hop reasoning experiment.

Tests vanilla, CoT, and Patchscope methods on multiple examples.

Usage:
    python scripts/run_multi_hop.py data/multihop_examples.json
    python scripts/run_multi_hop.py data/multihop_examples.json --layers 0,1,2,3,4,5,6,7,8,9 --limit 10
"""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import numpy as np

from patchscopes.model import load_model
from patchscopes.data import load_multihop_examples
from patchscopes.experiments.multi_hop import run_vanilla_and_patchscope, run_cot_phase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-hop Reasoning Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        'data_file',
        type=str,
        help='Path to multi-hop examples (JSON)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Base model name or path (for vanilla and patchscope methods)'
    )

    parser.add_argument(
        '--cot-model',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='Instruct model for Chain-of-Thought prompting'
    )

    parser.add_argument(
        '--layers',
        type=str,
        default='0,1,2,3,4,5,6,7,8,9,10,11',
        help='Comma-separated layer indices'
    )

    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=120,
        help='Maximum tokens to generate'
    )

    parser.add_argument(
        '--filter-solvable',
        action='store_true',
        default=False,
        help='Filter to only examples where both hops are individually solvable'
    )

    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Disable filtering (test all examples)'
    )

    parser.add_argument(
        '--output-plot',
        type=str,
        default='multihop_results.png',
        help='Output path for plot'
    )

    parser.add_argument(
        '--output-json',
        type=str,
        default='multihop_results.json',
        help='Output path for JSON results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of examples (for testing)'
    )

    return parser.parse_args()


def plot_results(results, output_path):
    """
    Create comparison plot: Accuracy by method.

    Shows:
    - Vanilla baseline (horizontal line)
    - CoT baseline (horizontal line)
    - Patchscope by layer (line plot)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Vanilla and CoT as horizontal lines
    ax.axhline(
        y=results['vanilla']['accuracy'],
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Vanilla (acc={results['vanilla']['accuracy']:.3f})"
    )

    ax.axhline(
        y=results['cot']['accuracy'],
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f"CoT (acc={results['cot']['accuracy']:.3f})"
    )

    # Patchscope by layer
    layers = sorted(results['patchscope']['accuracy_by_layer'].keys())
    accuracies = [results['patchscope']['accuracy_by_layer'][l] for l in layers]

    ax.plot(
        layers, accuracies,
        marker='o',
        linewidth=2,
        markersize=8,
        color='blue',
        label='Patchscope'
    )

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Multi-Hop Reasoning: Method Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12, loc='lower right')

    # Highlight best patchscope layer
    if 'best_layer' in results['patchscope']:
        best_layer = results['patchscope']['best_layer']
        best_acc = results['patchscope']['best_accuracy']

        ax.annotate(
            f'Best: Layer {best_layer}\n(acc={best_acc:.3f})',
            xy=(best_layer, best_acc),
            xytext=(best_layer, best_acc + 0.1),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")


def save_json_results(results, output_path):
    """Save detailed results to JSON."""
    # Convert any non-serializable types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj

    results_serializable = convert(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to {output_path}")


def print_sample_results(results, num_samples=3):
    """Print sample generations for qualitative analysis."""
    print("\n" + "=" * 80)
    print("Sample Results")
    print("=" * 80)

    for i, example in enumerate(results['vanilla']['per_example'][:num_samples]):
        print(f"\nExample {i+1}: {example['question']}")
        print(f"Target answer: {example['target_answer']}")
        print("-" * 40)
        print(f"Vanilla: {example['vanilla']['generated'][:80]}... [{'✓' if example['vanilla']['correct'] else '✗'}]")
        print(f"CoT: {example['cot']['generated'][:80]}... [{'✓' if example['cot']['correct'] else '✗'}]")
        print("Patchscope (first 3 layers):")
        for layer in sorted(example['patchscope_by_layer'].keys())[:3]:
            gen = example['patchscope_by_layer'][layer]['generated'][:80]
            correct = example['patchscope_by_layer'][layer]['correct']
            print(f"  Layer {layer}: {gen}... [{'✓' if correct else '✗'}]")
        print()


def main():
    args = parse_args()

    print("=" * 80)
    print("Multi-Hop Reasoning Experiment")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading examples from {args.data_file}...")
    examples = load_multihop_examples(args.data_file)
    print(f"Loaded {len(examples)} examples")

    if args.limit:
        examples = examples[:args.limit]
        print(f"Limited to {len(examples)} examples")
    print()

    # Load base model
    device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model: {args.model}")
    model = load_model(args.model, device=device, dtype=dtype)
    print(f"Base model loaded! ({model.cfg.n_layers} layers)")
    print()

    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(',')]
    print(f"Testing layers: {layers}")
    print()

    # Run experiment in two phases to manage memory
    filter_solvable = args.filter_solvable and not args.no_filter

    print("=" * 80)
    print("Phase 1: Vanilla + Patchscope (Base Model)")
    print("=" * 80)
    print()

    # Phase 1: Run vanilla and patchscope with base model
    results, filtered_examples = run_vanilla_and_patchscope(
        model,
        examples,
        layers,
        max_new_tokens=args.max_new_tokens,
        filter_solvable=filter_solvable
    )

    # Check if we need a separate CoT model
    use_instruct_format = args.cot_model != args.model

    print()
    print("=" * 80)
    print("Phase 2: Chain-of-Thought (Instruct Model)")
    print("=" * 80)
    print()

    if use_instruct_format:
        # Free base model from memory
        print("Freeing base model from memory...")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Load instruct model for CoT
        print(f"Loading instruct model for CoT: {args.cot_model}")
        cot_model = load_model(args.cot_model, device=device, dtype=dtype)
        print(f"Instruct model loaded! ({cot_model.cfg.n_layers} layers)")
        print()
    else:
        print("Using same model for CoT")
        cot_model = model

    # Phase 2: Run CoT with instruct model
    results = run_cot_phase(
        cot_model,
        results,
        filtered_examples,
        max_new_tokens=args.max_new_tokens,
        use_instruct_format=use_instruct_format
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print(f"Examples tested: {results['num_examples']}")
    if filter_solvable:
        print(f"Examples filtered out: {results['num_filtered']}")
    print()
    print(f"Vanilla accuracy: {results['vanilla']['accuracy']:.3f} ({results['vanilla']['correct_count']}/{results['num_examples']})")
    print(f"CoT accuracy: {results['cot']['accuracy']:.3f} ({results['cot']['correct_count']}/{results['num_examples']})")
    print()
    print("Patchscope accuracy by layer:")
    for layer in sorted(results['patchscope']['accuracy_by_layer'].keys()):
        acc = results['patchscope']['accuracy_by_layer'][layer]
        print(f"  Layer {layer}: {acc:.3f}")
    print()
    print(f"Best Patchscope: Layer {results['patchscope']['best_layer']} (acc={results['patchscope']['best_accuracy']:.3f})")
    print()

    # Print samples
    print_sample_results(results, num_samples=3)

    # Save results
    plot_results(results, args.output_plot)
    save_json_results(results, args.output_json)

    print()
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
