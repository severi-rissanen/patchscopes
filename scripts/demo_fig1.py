#!/usr/bin/env python3
"""
Demo script that reproduces Figure 1 from the Patchscopes paper.

This demonstrates the core patchscoping idea:
1. Extract a representation of a token from a source prompt (e.g., "CEO")
2. Patch it into an identity-style target prompt
3. Decode what the model thinks that token means by predicting next token(s)
"""
import sys
import os

# Add parent directory to path so we can import patchscopes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from patchscopes.model import load_model
from patchscopes.prompts import build_identity_prompt
from patchscopes.positions import find_substring_token_position, find_token_position, verify_single_token
from patchscopes.patch import extract_representation, run_with_patch, get_top_tokens


def main():
    print("=" * 80)
    print("Patchscopes Demo - Reproducing Figure 1")
    print("=" * 80)
    print()

    # Configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # or "meta-llama/Meta-Llama-3-8B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print()

    # Load model
    model = load_model(MODEL_NAME, device=DEVICE, dtype=DTYPE)

    print("Model loaded successfully!")
    print(f"Number of layers: {model.cfg.n_layers}")
    print()

    # Example from Figure 1
    source_text = "Amazon 's former CEO attended Oscars"
    target_word = "CEO"  # The token we want to decode

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
    print("Running Patchscope across layers...")
    print("=" * 80)
    print()

    # Test a few representative layers
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

        # Step 2: Patch into target and get logits
        logits = run_with_patch(
            model,
            target_prompt,
            layer=layer,
            target_position=placeholder_pos,
            source_vec=source_vec
        )

        # Step 3: Decode - get top predictions after the placeholder
        # We want predictions after "->", which is the position after placeholder
        prediction_pos = placeholder_pos + 1  # Position of "->"
        if prediction_pos >= logits.shape[1]:
            prediction_pos = placeholder_pos

        top_tokens = get_top_tokens(model, logits, prediction_pos, k=5)

        print(f"Top 5 predictions (decoding what '{target_word}' means):")
        for i, (token, prob) in enumerate(top_tokens, 1):
            print(f"  {i}. '{token}' (p={prob:.4f})")
        print()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print()
    print("Expected behavior:")
    print("- Early layers: predictions may be incoherent or generic")
    print("- Middle/late layers: predictions should relate to 'CEO' meaning")
    print("  (e.g., 'Jeff', 'Bezos', 'executive', 'chief', etc.)")


if __name__ == "__main__":
    main()
