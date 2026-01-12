#!/usr/bin/env python3
"""
Interactive multi-hop reasoning demonstration.

Shows how vanilla, CoT, and Patchscope methods solve a single 2-hop question
across different layers.

Usage:
    python scripts/run_multi_hop_interactive.py --example-id 0
    python scripts/run_multi_hop_interactive.py --example-id 3 --layer-range 0-15
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from patchscopes.model import load_model
from patchscopes.data import load_multihop_examples
from patchscopes.prompts import (
    build_hop1_prompt,
    build_hop2_prompt,
    build_vanilla_multihop_prompt,
    build_cot_multihop_prompt
)
from patchscopes.positions import find_substring_token_position, find_token_position, verify_single_token
from patchscopes.patch import extract_representation, generate_with_patch, vanilla_generate
from patchscopes.metrics import answer_appears_in_generation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Multi-hop Reasoning Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--example-id",
        type=int,
        default=0,
        help="Index of example to use from data file"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="data/multihop_examples.json",
        help="Path to multi-hop examples file"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model name or path"
    )

    parser.add_argument(
        "--layer-range",
        type=str,
        default="0-9",
        help="Layer range to test (e.g., '0-9' or '0,3,6,9')"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Multi-Hop Reasoning Interactive Demo")
    print("=" * 80)
    print()

    # Load examples
    print(f"Loading examples from {args.data_file}...")
    examples = load_multihop_examples(args.data_file)
    print(f"Loaded {len(examples)} examples")
    print()

    # Get specific example
    if args.example_id >= len(examples):
        print(f"Error: Example ID {args.example_id} not found (max: {len(examples)-1})")
        sys.exit(1)

    example = examples[args.example_id]

    print(f"Example {args.example_id}: {example['combined_question']}")
    print(f"Hop 1: {example['hop1_prompt']} -> {example['hop1_answer']}")
    print(f"Hop 2: {example['hop2_prompt_prefix']} {example['hop1_answer']} -> {example['hop2_answer']}")
    print(f"Target final answer: {example['hop2_answer']}")
    print()

    # Load model
    device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device=device, dtype=dtype)
    print(f"Model loaded! ({model.cfg.n_layers} layers)")
    print()

    # Parse layers
    if '-' in args.layer_range:
        start, end = args.layer_range.split('-')
        layers = list(range(int(start), min(int(end)+1, model.cfg.n_layers)))
    else:
        layers = [int(x) for x in args.layer_range.split(',')]

    print(f"Testing layers: {layers}")
    print()

    # Find placeholder
    placeholder = " x"
    if not verify_single_token(model, placeholder):
        for alt in [" X", " z", " #"]:
            if verify_single_token(model, alt):
                placeholder = alt
                break

    print(f"Using placeholder: '{placeholder}'")
    print()

    # Method 1: Vanilla
    print("=" * 80)
    print("Method 1: Vanilla (No Chain-of-Thought)")
    print("=" * 80)
    vanilla_prompt = build_vanilla_multihop_prompt(example['combined_question'])
    print(f"Prompt: {vanilla_prompt}")

    vanilla_tokens, _ = vanilla_generate(model, vanilla_prompt, max_new_tokens=args.max_new_tokens)
    vanilla_text = "".join(vanilla_tokens)
    vanilla_correct = answer_appears_in_generation(vanilla_tokens, example['hop2_answer'])

    print(f"Generated: {vanilla_text}")
    print(f"Correct: {'✓' if vanilla_correct else '✗'}")
    print()

    # Method 2: CoT
    print("=" * 80)
    print("Method 2: Chain-of-Thought")
    print("=" * 80)
    cot_prompt = build_cot_multihop_prompt(example['combined_question'])
    print(f"Prompt: {cot_prompt}")

    cot_tokens, _ = vanilla_generate(model, cot_prompt, max_new_tokens=args.max_new_tokens)
    cot_text = "".join(cot_tokens)
    cot_correct = answer_appears_in_generation(cot_tokens, example['hop2_answer'])

    print(f"Generated: {cot_text}")
    print(f"Correct: {'✓' if cot_correct else '✗'}")
    print()

    # Method 3: Patchscope
    print("=" * 80)
    print("Method 3: Patchscope (Layer Sweep)")
    print("=" * 80)

    hop1_prompt = build_hop1_prompt(example['hop1_prompt'], placeholder)
    hop2_prompt = build_hop2_prompt(example['hop2_prompt_prefix'], placeholder)

    print(f"Hop 1 prompt: {hop1_prompt}")
    print(f"Hop 2 prompt: {hop2_prompt}")
    print()

    # Find positions
    hop1_position = find_substring_token_position(model, hop1_prompt, "->")
    if hop1_position is None:
        hop1_position = len(model.to_tokens(hop1_prompt, prepend_bos=True)[0]) - 1

    hop2_placeholder_pos = find_token_position(model, hop2_prompt, placeholder)

    print(f"Extracting from hop1 position: {hop1_position}")
    print(f"Patching into hop2 position: {hop2_placeholder_pos}")
    print()

    # Track results for summary
    patchscope_results = []

    for layer in layers:
        print(f"Layer {layer}:")
        print("-" * 40)

        # Extract and patch
        with torch.no_grad():
            bridge_vec = extract_representation(model, hop1_prompt, layer=layer, position=hop1_position)
            patch_tokens, _ = generate_with_patch(
                model, hop2_prompt,
                target_position=hop2_placeholder_pos,
                source_vec=bridge_vec,
                max_new_tokens=args.max_new_tokens
            )

        patch_text = "".join(patch_tokens)
        patch_correct = answer_appears_in_generation(patch_tokens, example['hop2_answer'])

        print(f"  Generated: {patch_text}")
        print(f"  Correct: {'✓' if patch_correct else '✗'}")
        print()

        patchscope_results.append((layer, patch_correct))

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Vanilla: {'✓' if vanilla_correct else '✗'}")
    print(f"CoT: {'✓' if cot_correct else '✗'}")
    print()
    print("Patchscope by layer:")
    for layer, correct in patchscope_results:
        print(f"  Layer {layer}: {'✓' if correct else '✗'}")
    print()

    # Count successes
    patchscope_success_count = sum(1 for _, correct in patchscope_results if correct)
    print(f"Patchscope success rate: {patchscope_success_count}/{len(layers)} layers")
    print()


if __name__ == "__main__":
    main()
