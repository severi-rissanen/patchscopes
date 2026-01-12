#!/usr/bin/env python3
"""
Run entity resolution experiments across layers.
Evaluates when/how model resolves entity meanings (Section 4.3).
"""
import argparse
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import numpy as np

from patchscopes.model import load_model
from patchscopes.data import load_entities, validate_entities
from patchscopes.experiments.entity import run_entity_resolution_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entity Resolution Experiments - Description generation across layers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'entity_file',
        type=str,
        help='Path to entity data file (JSON or CSV)',
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Model name or path',
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='0,1,2,3,4,5,6,7,8,9',
        help='Comma-separated layer indices to test (e.g., "0,1,2,3,4,5,6,7,8,9")',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=20,
        help='Maximum tokens to generate per entity description',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Generation temperature (0 = greedy decoding)',
    )
    parser.add_argument(
        '--num-demos',
        type=int,
        default=3,
        help='Number of few-shot examples in target prompt',
    )
    parser.add_argument(
        '--output-plot',
        type=str,
        default='entity_results.png',
        help='Output path for plot',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='entity_results.json',
        help='Output path for detailed JSON results',
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
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of entities to process (for testing)',
    )
    return parser.parse_args()


def plot_results(results, output_path):
    """
    Create entity resolution plot: ROUGE-L vs Layer.

    Single plot showing how entity resolution quality improves across layers.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    layers = results['layers']
    rouge_scores = results['rouge_l_scores']

    # Main line plot
    ax.plot(layers, rouge_scores, marker='o', linewidth=2, markersize=8,
            color='#2E86AB', label='ROUGE-L Score')

    # Add confidence band if we have per-entity data
    if 'per_entity_scores' in results:
        # Calculate std dev at each layer
        std_devs = []
        for layer in layers:
            layer_scores = []
            for entity_result in results['per_entity_scores']:
                if layer in entity_result['scores_by_layer']:
                    layer_scores.append(entity_result['scores_by_layer'][layer])

            if layer_scores:
                std_devs.append(np.std(layer_scores))
            else:
                std_devs.append(0.0)

        # Plot confidence band (mean ± std)
        rouge_scores_array = np.array(rouge_scores)
        std_devs_array = np.array(std_devs)

        ax.fill_between(layers,
                        rouge_scores_array - std_devs_array,
                        rouge_scores_array + std_devs_array,
                        alpha=0.2, color='#2E86AB')

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('ROUGE-L Score', fontsize=14)
    ax.set_title('Entity Resolution Quality Across Layers', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)

    # Add annotation for best layer
    if rouge_scores:
        best_layer_idx = np.argmax(rouge_scores)
        best_layer = layers[best_layer_idx]
        best_score = rouge_scores[best_layer_idx]

        ax.annotate(
            f'Best: Layer {best_layer}\n(ROUGE-L = {best_score:.3f})',
            xy=(best_layer, best_score),
            xytext=(best_layer, best_score + 0.1),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")


def save_json_results(results, output_path):
    """
    Save detailed results to JSON file.

    Includes per-entity scores and generated text for analysis.
    """
    # Convert numpy types to native Python for JSON serialization
    serializable_results = {
        'layers': [int(l) for l in results['layers']],
        'rouge_l_scores': [float(s) for s in results['rouge_l_scores']],
        'per_entity_scores': []
    }

    for entity_result in results['per_entity_scores']:
        serializable_results['per_entity_scores'].append({
            'entity': entity_result['entity'],
            'reference': entity_result['reference'],
            'scores_by_layer': {
                int(k): float(v) for k, v in entity_result['scores_by_layer'].items()
            },
            'generations_by_layer': {
                int(k): str(v) for k, v in entity_result['generations_by_layer'].items()
            }
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Saved detailed results to {output_path}")


def print_sample_generations(results, num_samples=3):
    """
    Print sample entity generations for qualitative inspection.
    """
    if not results['per_entity_scores']:
        return

    print("\n" + "=" * 80)
    print("Sample Entity Generations")
    print("=" * 80)

    # Show first few entities
    for i, entity_result in enumerate(results['per_entity_scores'][:num_samples]):
        print(f"\nEntity: {entity_result['entity']}")
        print(f"Reference: {entity_result['reference']}")
        print("-" * 40)

        for layer in results['layers']:
            if layer in entity_result['generations_by_layer']:
                gen_text = entity_result['generations_by_layer'][layer]
                score = entity_result['scores_by_layer'][layer]
                print(f"  Layer {layer:2d} (ROUGE-L={score:.3f}): {gen_text}")
        print()


def main():
    args = parse_args()

    print("=" * 80)
    print("Entity Resolution Experiment")
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
    print(f"  Entity file: {args.entity_file}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Random seed: {args.seed}")
    print()

    # Load entities
    print("Loading entity data...")
    try:
        entities = load_entities(args.entity_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading entity file: {e}")
        sys.exit(1)

    print(f"Loaded {len(entities)} entities")

    # Validate entities
    print("Validating entity data...")
    valid_count = validate_entities(entities)
    print(f"Validated {valid_count}/{len(entities)} entities")
    print()

    # Apply limit if specified
    if args.limit:
        entities = entities[:args.limit]
        print(f"Limited to first {len(entities)} entities")
        print()

    # Load model
    print(f"Loading model...")
    model = load_model(args.model, device=args.device, dtype=dtype)
    print(f"Model loaded! ({model.cfg.n_layers} layers)")
    print()

    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(',')]

    # Validate layers
    invalid_layers = [l for l in layers if l >= model.cfg.n_layers]
    if invalid_layers:
        print(f"Error: Invalid layers {invalid_layers} for model with {model.cfg.n_layers} layers")
        sys.exit(1)

    print(f"Testing layers: {layers}")
    print()

    # Run experiment
    print("=" * 80)
    print("Running Experiment")
    print("=" * 80)
    print()

    results = run_entity_resolution_experiment(
        model,
        entities,
        layers,
        num_demos=args.num_demos,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print(f"Layers: {results['layers']}")
    print(f"ROUGE-L scores: {[f'{x:.3f}' for x in results['rouge_l_scores']]}")
    print()

    # Find best layer
    if results['rouge_l_scores']:
        best_idx = np.argmax(results['rouge_l_scores'])
        best_layer = results['layers'][best_idx]
        best_score = results['rouge_l_scores'][best_idx]
        print(f"Best performing layer: {best_layer} (ROUGE-L = {best_score:.3f})")
        print()

    # Print sample generations
    print_sample_generations(results, num_samples=3)

    # Save results
    plot_results(results, args.output_plot)
    save_json_results(results, args.output_json)

    print()
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print()
    print("Key findings to look for:")
    print("- Early layers (0-2): Often produce generic descriptions")
    print("- Middle layers (3-6): Begin capturing entity category")
    print("- Later layers (7-9): Generate accurate, specific descriptions")
    print()


if __name__ == '__main__':
    main()
