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
        help="Base model name or path (for vanilla and patchscope methods)"
    )

    parser.add_argument(
        "--cot-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B", # this should be the Instruct model really, but for memory reasons keeping like this
        help="Instruct model for Chain-of-Thought prompting"
    )

    parser.add_argument(
        "--layer-range",
        type=str,
        default="0-20",
        help="Layer range to test (e.g., '0-9' or '0,3,6,9')"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
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

    # Load base model
    device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model: {args.model}")
    model = load_model(args.model, device=device, dtype=dtype)
    print(f"Base model loaded! ({model.cfg.n_layers} layers)")
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

    # Method 1: Vanilla (using base model)
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

    # Method 2: Patchscope (using base model)
    print("=" * 80)
    print("Method 2: Patchscope (Layer Sweep)")
    print("=" * 80)

    hop1_prompt = build_hop1_prompt(example['hop1_prompt'], placeholder)
    hop2_prompt = build_hop2_prompt(example['hop2_prompt_prefix'], placeholder)

    print(f"Hop 1 prompt: {hop1_prompt}")
    hop1_tokens = model.to_tokens(hop1_prompt, prepend_bos=True)[0]
    token_texts = [model.to_string(hop1_tokens[:i+1])[-len(model.to_string(hop1_tokens[i])):] for i in range(len(hop1_tokens))]
    print("Hop 1 prompt, tokenized:")
    for idx, (tok_id, tok_text) in enumerate(zip(hop1_tokens, token_texts)):
        print(f"  Token {idx}: id={tok_id}, text='{tok_text}'")
    print(f"Hop 2 prompt: {hop2_prompt}")
    hop2_tokens = model.to_tokens(hop2_prompt, prepend_bos=True)[0]
    hop2_token_texts = [model.to_string(hop2_tokens[:i+1])[-len(model.to_string(hop2_tokens[i])):] for i in range(len(hop2_tokens))]
    print("Hop 2 prompt, tokenized:")
    for idx, (tok_id, tok_text) in enumerate(zip(hop2_tokens, hop2_token_texts)):
        print(f"  Token {idx}: id={tok_id}, text='{tok_text}'")
    print()

    # Find positions - extract from the last token of the hop1 query phrase
    hop1_position = find_substring_token_position(model, hop1_prompt, example['hop1_prompt'], return_last=True)
    if hop1_position is None:
        print(f"Warning: Could not find '{example['hop1_prompt']}' in hop1_prompt, using last token")
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
            # patch_tokens, _ = generate_with_patch(model, hop2_prompt, target_position=hop2_placeholder_pos, source_vec=bridge_vec, max_new_tokens=args.max_new_tokens)

        patch_text = "".join(patch_tokens)
        patch_correct = answer_appears_in_generation(patch_tokens, example['hop2_answer'])

        print(f"  Generated: {patch_text}")
        print(f"  Correct: {'✓' if patch_correct else '✗'}")
        print()

        patchscope_results.append((layer, patch_correct))

    # Method 3: CoT (using instruct model)
    # Free base model and load instruct model if different
    use_instruct_format = args.cot_model != args.model
    if use_instruct_format:
        print("=" * 80)
        print("Freeing base model from memory...")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Loading instruct model for CoT: {args.cot_model}")
        cot_model = load_model(args.cot_model, device=device, dtype=dtype)
        print(f"Instruct model loaded! ({cot_model.cfg.n_layers} layers)")
        print()
    else:
        print("Using same model for CoT")
        cot_model = model
        print()

    print("=" * 80)
    print("Method 3: Chain-of-Thought (Instruct Model)")
    print("=" * 80)
    cot_prompt = build_cot_multihop_prompt(example['combined_question'], use_instruct_format=use_instruct_format)
    print(f"Prompt: {cot_prompt[:200]}..." if len(cot_prompt) > 200 else f"Prompt: {cot_prompt}")

    cot_tokens, _ = vanilla_generate(cot_model, cot_prompt, max_new_tokens=args.max_new_tokens)
    cot_text = "".join(cot_tokens)
    cot_correct = answer_appears_in_generation(cot_tokens, example['hop2_answer'])

    print(f"Generated: {cot_text}")
    print(f"Correct: {'✓' if cot_correct else '✗'}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Vanilla: {'✓' if vanilla_correct else '✗'}")
    print(f"Patchscope by layer:")
    for layer, correct in patchscope_results:
        print(f"  Layer {layer}: {'✓' if correct else '✗'}")
    print(f"CoT: {'✓' if cot_correct else '✗'}")
    print()

    # Count successes
    patchscope_success_count = sum(1 for _, correct in patchscope_results if correct)
    print(f"Patchscope success rate: {patchscope_success_count}/{len(layers)} layers")
    print()


if __name__ == "__main__":
    main()
