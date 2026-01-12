#!/usr/bin/env python3
"""
Demo script that reproduces Figure 1 from the Patchscopes paper.

This demonstrates the core patchscoping idea:
1. Extract a representation of a token from a source prompt (e.g., "CEO")
2. Patch it into an identity-style target prompt
3. Decode what the model thinks that token means by predicting next token(s)
"""
import argparse
import sys
import os

# Add parent directory to path so we can import patchscopes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from patchscopes.model import load_model
from patchscopes.prompts import build_identity_prompt
from patchscopes.positions import find_substring_token_position, find_token_position, verify_single_token
from patchscopes.patch import (
    extract_representation,
    generate_with_patch,
    run_with_patch,
    get_top_tokens,
)


def parse_layers(value):
    """Parse comma-separated layer indices into a list of integers."""
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid layers format: '{value}'. Expected comma-separated integers (e.g., '0,5,10,15')"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patchscopes Demo - Reproducing Figure 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model name or path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "top_k"],
        default="generate",
        help="Decoding mode: 'generate' for autoregressive, 'top_k' for top-k predictions",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
        help="Number of tokens to generate (for 'generate' mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tokens to show (for 'top_k' mode)",
    )
    parser.add_argument(
        "--source-text",
        type=str,
        default="Amazon 's former CEO attended Oscars",
        help="Source text to extract representation from",
    )
    parser.add_argument(
        "--target-word",
        type=str,
        default="CEO",
        help="Target word/token to decode",
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default="0,5,10,20,25,27,31",
        help="Comma-separated layer indices to test (e.g., '0,5,10,15')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Patchscopes Demo - Reproducing Figure 1")
    print("=" * 80)
    print()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    print()

    # Load model
    model = load_model(args.model, device=device, dtype=dtype)

    print("Model loaded successfully!")
    print(f"Number of layers: {model.cfg.n_layers}")
    print()

    # Source text and target word
    source_text = args.source_text
    target_word = args.target_word

    print(f"Source text: '{source_text}'")
    print(f"Target word to decode: '{target_word}'")
    print()

    # Find the position of "CEO" in the source text
    ceo_position = find_substring_token_position(model, source_text, target_word)

    if ceo_position is None:
        print(f"Error: Could not find '{target_word}' in source text")
        return

    # Show tokenization
    tokens = model.to_tokens(source_text, prepend_bos=True)
    print(f"Source tokenization:")
    for i in range(tokens.shape[1]):
        token_str = model.to_string(tokens[0, i])
        marker = " <-- TARGET" if i == ceo_position else ""
        print(f"  Position {i}: '{token_str}'{marker}")
    print()

    # Build identity-style target prompt
    placeholder = " x"

    # Verify placeholder is single token
    if not verify_single_token(model, placeholder):
        print(f"Warning: '{placeholder}' is not a single token!")
        print("Trying ' X' instead...")
        placeholder = " X"
        if not verify_single_token(model, placeholder):
            print("Error: Could not find single-token placeholder")
            return

    target_prompt = build_identity_prompt(placeholder=placeholder, num_demos=5)
    print(f"Target (identity) prompt: '{target_prompt}'")
    print()

    # Find placeholder position in target
    placeholder_pos = find_token_position(model, target_prompt, placeholder)
    if placeholder_pos is None:
        print(f"Error: Could not find placeholder '{placeholder}' in target prompt")
        return

    print(f"Placeholder position in target: {placeholder_pos}")
    print()

    # Run patchscope across multiple layers
    print("=" * 80)
    print(f"Running Patchscope across layers (mode: {args.mode})...")
    print("=" * 80)
    print()

    # Determine which layers to test
    if args.layers is not None:
        layers_to_test = [l for l in args.layers if l < model.cfg.n_layers]
    else:
        layers_to_test = [0, 5, 10, 15, 20, 25, 31]  # Llama-3-8B has 32 layers (0-31)
        layers_to_test = [l for l in layers_to_test if l < model.cfg.n_layers]

    for layer in layers_to_test:
        print(f"Layer {layer}:")
        print("-" * 40)

        # Step 1: Extract representation from source
        source_vec = extract_representation(
            model,
            source_text,
            layer=layer,
            position=ceo_position
        )

        if args.mode == "generate":
            # Step 2a: Patch at layer 0 and generate tokens autoregressively
            generated_tokens, _ = generate_with_patch(
                model,
                target_prompt,
                target_position=placeholder_pos,
                source_vec=source_vec,
                max_new_tokens=args.max_new_tokens,
                temperature=0,  # greedy decoding
            )

            # Step 3a: Display the generated output
            generated_text = "".join(generated_tokens)
            print(f"Generated (decoding what '{target_word}' means): {generated_text}")
            print(f"  Tokens: {generated_tokens}")

        else:  # top_k mode
            # Step 2b: Patch at layer 0 and get logits
            logits = run_with_patch(
                model,
                target_prompt,
                target_position=placeholder_pos,
                source_vec=source_vec
            )

            # Step 3b: Decode - get top predictions after the placeholder
            prediction_pos = placeholder_pos + 1  # Position after placeholder
            if prediction_pos >= logits.shape[1]:
                prediction_pos = placeholder_pos

            top_tokens = get_top_tokens(model, logits, prediction_pos, k=args.top_k)

            print(f"Top {args.top_k} predictions (decoding what '{target_word}' means):")
            for i, (token, prob) in enumerate(top_tokens, 1):
                print(f"  {i}. '{token}' (p={prob:.4f})")

        print()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print()
    print("Expected behavior:")
    print("- Early layers: predictions may be incoherent or generic")
    print("- Middle/late layers: should relate to 'CEO' meaning")
    print("  (e.g., 'Jeff', 'Bezos', 'executive', 'chief', etc.)")


if __name__ == "__main__":
    main()
