#!/usr/bin/env python3
"""
Interactive entity resolution using Patchscopes.

Takes an entity name and generates descriptions across layers to visualize
when/how the model resolves the entity's meaning (Section 4.3 of paper).

Usage:
    python scripts/run_entity_interactive.py "Diana, Princess of Wales"
    python scripts/run_entity_interactive.py "Albert Einstein" --layer-range 0-15
"""
import argparse
import sys
import os

# Add parent directory to path so we can import patchscopes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from patchscopes.model import load_model
from patchscopes.prompts import build_entity_description_prompt
from patchscopes.positions import (
    find_substring_token_position,
    find_token_position,
    verify_single_token,
)
from patchscopes.patch import extract_representation, generate_with_patch


# Hardcoded few-shot examples covering diverse entity types
ENTITY_EXAMPLES = [
    ("Syria", "Country in the Middle East"),
    ("Leonardo DiCaprio", "American actor and film producer"),
    ("Samsung", "South Korean multinational electronics company"),
    ("Mount Everest", "Highest mountain in the world"),
    ("World War II", "Global war that lasted from 1939 to 1945"),
]


def parse_layer_range(value, max_layers):
    """
    Parse layer range specification.

    Supports:
    - Range format: "0-9" -> [0, 1, 2, ..., 9]
    - List format: "0,5,10,15" -> [0, 5, 10, 15]

    Args:
        value: String specifying layer range
        max_layers: Maximum number of layers in model

    Returns:
        Sorted list of layer indices
    """
    if '-' in value and ',' not in value:
        # Range format: "0-9"
        start, end = value.split('-')
        start, end = int(start.strip()), int(end.strip())
        layers = list(range(start, min(end + 1, max_layers)))
    else:
        # Comma-separated format: "0,5,10"
        layers = [int(x.strip()) for x in value.split(',')]
        layers = [l for l in layers if l < max_layers]

    return sorted(layers)


def find_valid_placeholder(model):
    """
    Find a valid single-token placeholder.

    Args:
        model: HookedTransformer model

    Returns:
        Valid single-token placeholder string
    """
    candidates = [" x", " X", " z", " Z", " y", " Y"]

    for placeholder in candidates:
        if verify_single_token(model, placeholder):
            return placeholder

    # Fallback: warn and use " x" anyway
    print("WARNING: Could not find single-token placeholder, using ' x'")
    return " x"


def display_tokenization(model, text, target_position=None):
    """
    Display tokenization with optional highlighting of target position.

    Args:
        model: HookedTransformer model
        text: Text to tokenize
        target_position: Optional position to highlight
    """
    tokens = model.to_tokens(text, prepend_bos=True)
    print("Source tokenization:")
    for i in range(tokens.shape[1]):
        token_str = model.to_string(tokens[0, i])
        marker = " <-- SOURCE (last token)" if i == target_position else ""
        print(f"  Position {i}: '{token_str}'{marker}")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Entity Resolution using Patchscopes (Section 4.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "entity",
        type=str,
        help="Entity name to resolve (e.g., 'Diana, Princess of Wales')",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model name or path",
    )

    parser.add_argument(
        "--layer-range",
        type=str,
        default="0-9",
        help="Layer range to test (e.g., '0-9' for range, '0,5,10' for specific layers)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate per description",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0 = greedy decoding)",
    )

    parser.add_argument(
        "--num-demos",
        type=int,
        default=3,
        help="Number of few-shot examples in target prompt",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )

    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()

    # Print header
    print("=" * 80)
    print(f"Entity Resolution: {args.entity}")
    print("=" * 80)
    print()

    # Validate entity name
    if not args.entity.strip():
        print("Error: Entity name cannot be empty")
        sys.exit(1)

    # Setup device and dtype
    device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load model
    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    print()

    model = load_model(args.model, device=device, dtype=dtype)

    print("Model loaded successfully!")
    print(f"Number of layers: {model.cfg.n_layers}")
    print()

    # Source text is just the entity name
    source_text = args.entity

    # Find entity position in source
    entity_position = find_substring_token_position(model, source_text, args.entity)

    if entity_position is None:
        print(f"Error: Could not locate entity '{args.entity}' in tokenized source")
        print("This may occur with unusual characters or tokenization edge cases")
        sys.exit(1)

    # Check if single-token entity (warn but continue)
    tokens = model.to_tokens(source_text, prepend_bos=True)
    if tokens.shape[1] <= 2:  # BOS + 1 token
        print(f"Warning: Entity tokenizes to only 1 token. Results may be less interpretable.")
        print()

    # Display source tokenization
    display_tokenization(model, source_text, target_position=entity_position)

    # Build target prompt
    selected_examples = ENTITY_EXAMPLES[:args.num_demos]

    # Find valid placeholder
    placeholder = find_valid_placeholder(model)

    # Build entity description prompt
    target_prompt = build_entity_description_prompt(selected_examples, placeholder)

    print(f"Target prompt: \"{target_prompt}\"")
    print()

    # Find placeholder position in target
    placeholder_pos = find_token_position(model, target_prompt, placeholder)

    if placeholder_pos is None:
        print(f"Error: Could not find placeholder '{placeholder}' in target prompt")
        print("This is likely a tokenization issue with the chosen placeholder")
        sys.exit(1)

    print(f"Placeholder position: {placeholder_pos}")
    print()

    # Parse layer range
    try:
        layers = parse_layer_range(args.layer_range, model.cfg.n_layers)
    except (ValueError, IndexError) as e:
        print(f"Error: Invalid layer range '{args.layer_range}': {e}")
        sys.exit(1)

    if not layers:
        print(f"Error: No valid layers in range '{args.layer_range}'")
        print(f"Model has {model.cfg.n_layers} layers (0-{model.cfg.n_layers-1})")
        sys.exit(1)

    # Print configuration
    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Layers: {layers}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print()

    # Main layer sweep
    print("=" * 80)
    print("Generating Descriptions Across Layers")
    print("=" * 80)
    print()

    for layer in layers:
        print(f"Layer {layer}:")
        print("-" * 40)

        try:
            # Extract representation from source at entity position
            source_vec = extract_representation(
                model,
                source_text,
                layer=layer,
                position=entity_position
            )

            # Generate with patch at placeholder position
            generated_tokens, _ = generate_with_patch(
                model,
                target_prompt,
                target_position=placeholder_pos,
                source_vec=source_vec,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            # Display results
            generated_text = "".join(generated_tokens)
            print(f"Generated: {generated_text}")
            print(f"Tokens: {generated_tokens}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nError at layer {layer}: CUDA out of memory")
                print("Try using --device cpu or reducing --max-new-tokens")
                sys.exit(1)
            raise

        print()

    # Completion message
    print("=" * 80)
    print("Entity Resolution Complete!")
    print("=" * 80)
    print()
    print("Observations:")
    print("- Early layers (0-2): Often produce generic or incorrect descriptions")
    print("- Middle layers (3-6): Begin to capture entity category")
    print("- Later layers (7-9): Generate specific, accurate descriptions")
    print()


if __name__ == "__main__":
    main()
