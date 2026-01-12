"""Entity resolution experiment across layers (Section 4.3)."""
import torch
import numpy as np
from tqdm import tqdm

from patchscopes.metrics import rouge_l
from patchscopes.prompts import build_entity_description_prompt
from patchscopes.positions import find_substring_token_position, verify_single_token
from patchscopes.patch import extract_representation, generate_with_patch


# Few-shot examples for entity description prompt
DEFAULT_ENTITY_EXAMPLES = [
    ("Syria", "Country in the Middle East"),
    ("Leonardo DiCaprio", "American actor and film producer"),
    ("Samsung", "South Korean multinational electronics company"),
]


def run_entity_resolution_experiment(
    model,
    entities,
    layers,
    placeholder=" x",
    num_demos=3,
    max_new_tokens=20,
    temperature=0.0,
):
    """
    Run entity resolution experiment across layers.

    For each layer and entity:
    1. Extract hidden state from entity name at layer ℓ
    2. Patch into entity description prompt
    3. Generate description
    4. Compute ROUGE-L against reference description

    Args:
        model: HookedTransformer model
        entities: List of dicts with 'entity' and 'reference' keys
        layers: List of layer indices to evaluate
        placeholder: Single-token placeholder for target prompt
        num_demos: Number of few-shot examples in target prompt
        max_new_tokens: Maximum tokens to generate per entity
        temperature: Generation temperature (0 = greedy)

    Returns:
        results: Dict with structure:
            {
                'layers': [0, 1, 2, ...],
                'rouge_l_scores': [0.15, 0.23, ...],  # averaged across entities
                'per_entity_scores': [  # detailed per-entity results
                    {
                        'entity': 'Diana, Princess of Wales',
                        'reference': 'Member of British royal family',
                        'scores_by_layer': {0: 0.1, 1: 0.2, ...},
                        'generations_by_layer': {0: 'Princess', 1: 'British royal', ...}
                    },
                    ...
                ]
            }
    """
    # Verify placeholder is single token
    if not verify_single_token(model, placeholder):
        # Try alternatives
        for alt in [" X", " z", " #", " Y"]:
            if verify_single_token(model, alt):
                placeholder = alt
                print(f"Using placeholder: '{placeholder}'")
                break
        else:
            raise ValueError(
                f"Could not find single-token placeholder. "
                f"Tried: {placeholder}, ' X', ' z', ' #', ' Y'"
            )

    # Build target prompt once (reuse for all entities/layers)
    selected_examples = DEFAULT_ENTITY_EXAMPLES[:num_demos]
    target_prompt = build_entity_description_prompt(selected_examples, placeholder)

    # Find placeholder position
    placeholder_pos = find_substring_token_position(model, target_prompt, placeholder)

    if placeholder_pos is None:
        raise ValueError(
            f"Could not find placeholder '{placeholder}' in target prompt: '{target_prompt}'"
        )

    print(f"Target prompt: '{target_prompt}'")
    print(f"Placeholder position: {placeholder_pos}")
    print(f"Evaluating {len(entities)} entities across {len(layers)} layers")
    print()

    # Initialize result storage
    results = {
        'layers': layers,
        'rouge_l_scores': [],
        'per_entity_scores': [],
    }

    # Process each entity
    per_entity_data = []

    for entity_idx, entity_data in enumerate(tqdm(entities, desc="Processing entities")):
        entity_name = entity_data['entity']
        reference_desc = entity_data['reference']

        entity_results = {
            'entity': entity_name,
            'reference': reference_desc,
            'scores_by_layer': {},
            'generations_by_layer': {},
        }

        # Find entity position in source
        source_text = entity_name
        entity_position = find_substring_token_position(model, source_text, entity_name)

        if entity_position is None:
            print(f"\nWarning: Could not locate entity '{entity_name}' in tokenization, skipping")
            continue

        # Process each layer for this entity
        for layer in layers:
            try:
                # Extract representation from source at entity position
                with torch.no_grad():
                    source_vec = extract_representation(
                        model,
                        source_text,
                        layer=layer,
                        position=entity_position
                    )

                # Generate with patch at placeholder position
                with torch.no_grad():
                    generated_tokens, _ = generate_with_patch(
                        model,
                        target_prompt,
                        target_position=placeholder_pos,
                        source_vec=source_vec,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )

                # Combine generated tokens into text
                generated_text = "".join(generated_tokens).strip()

                # Handle empty generation
                if not generated_text:
                    generated_text = ""

                # Compute ROUGE-L score
                score = rouge_l(generated_text, reference_desc)

                # Store results
                entity_results['scores_by_layer'][layer] = score
                entity_results['generations_by_layer'][layer] = generated_text

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nError: CUDA out of memory at layer {layer}, entity {entity_idx}")
                    print("Consider using --device cpu or reducing --max-new-tokens")
                    raise
                else:
                    print(f"\nError processing entity '{entity_name}' at layer {layer}: {e}")
                    entity_results['scores_by_layer'][layer] = 0.0
                    entity_results['generations_by_layer'][layer] = ""

        per_entity_data.append(entity_results)

    results['per_entity_scores'] = per_entity_data

    # Aggregate ROUGE-L scores across entities for each layer
    for layer in layers:
        layer_scores = []

        for entity_result in per_entity_data:
            if layer in entity_result['scores_by_layer']:
                layer_scores.append(entity_result['scores_by_layer'][layer])

        if layer_scores:
            mean_score = np.mean(layer_scores)
            results['rouge_l_scores'].append(mean_score)
        else:
            results['rouge_l_scores'].append(0.0)

        # Print summary for this layer
        if layer_scores:
            print(f"Layer {layer}: ROUGE-L = {mean_score:.3f} "
                  f"(min={min(layer_scores):.3f}, max={max(layer_scores):.3f})")

    return results
