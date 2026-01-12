"""Multi-hop reasoning experiment with Patchscopes intervention."""
import torch
from tqdm import tqdm

from patchscopes.prompts import (
    build_hop1_prompt,
    build_hop2_prompt,
    build_vanilla_multihop_prompt,
    build_cot_multihop_prompt
)
from patchscopes.positions import find_substring_token_position, find_token_position, verify_single_token
from patchscopes.patch import extract_representation, generate_with_patch, vanilla_generate
from patchscopes.metrics import answer_appears_in_generation


def check_hop_solvability(
    model,
    hop_prompt,
    target_answer,
    max_new_tokens=20,
    temperature=0.0
):
    """
    Check if a single hop is solvable by the model.

    Args:
        model: HookedTransformer model
        hop_prompt: Prompt for this hop
        target_answer: Expected answer
        max_new_tokens: Number of tokens to generate
        temperature: Generation temperature

    Returns:
        is_solvable: Boolean indicating if answer appears
        generated_text: Generated text for debugging
    """
    generated_tokens, _ = vanilla_generate(
        model, hop_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    is_solvable = answer_appears_in_generation(generated_tokens, target_answer)
    generated_text = "".join(generated_tokens)

    return is_solvable, generated_text


def run_vanilla_and_patchscope(
    model,
    examples,
    layers,
    placeholder=" x",
    max_new_tokens=20,
    temperature=0.0,
    filter_solvable=False,
    check_n_tokens=20,
):
    """
    Run vanilla and patchscope experiments (Phase 1).

    Args:
        model: HookedTransformer model (base model)
        examples: List of multi-hop example dicts
        layers: List of layer indices to test for patchscope
        placeholder: Single-token placeholder
        max_new_tokens: Tokens to generate
        temperature: Generation temperature (0 = greedy)
        filter_solvable: Whether to pre-filter examples
        check_n_tokens: Tokens to generate for solvability checks

    Returns:
        results: Partial results dict (without CoT)
        filtered_examples: Examples used (for CoT phase)
    """
    # Verify placeholder
    if not verify_single_token(model, placeholder):
        for alt in [" X", " z", " #", " Y"]:
            if verify_single_token(model, alt):
                placeholder = alt
                print(f"Using placeholder: '{placeholder}'")
                break
        else:
            raise ValueError("Could not find single-token placeholder")

    # Filter examples if requested
    filtered_examples = []

    if filter_solvable:
        print(f"\nFiltering {len(examples)} examples for solvability...")

        for example in tqdm(examples, desc="Filtering"):
            # Check hop 1 solvability
            hop1_prompt = build_hop1_prompt(example['hop1_prompt'], placeholder)
            hop1_solvable, _ = check_hop_solvability(
                model, hop1_prompt, example['hop1_answer'],
                max_new_tokens=check_n_tokens, temperature=temperature
            )

            if not hop1_solvable:
                continue

            # Check hop 2 solvability (with explicit bridge entity)
            hop2_test_prompt = f"{example['hop2_prompt_prefix']} {example['hop1_answer']} is"
            hop2_solvable, _ = check_hop_solvability(
                model, hop2_test_prompt, example['hop2_answer'],
                max_new_tokens=check_n_tokens, temperature=temperature
            )

            if not hop2_solvable:
                continue

            filtered_examples.append(example)

        print(f"Filtered to {len(filtered_examples)}/{len(examples)} solvable examples")
    else:
        filtered_examples = examples

    if not filtered_examples:
        raise ValueError("No examples passed filtering!")

    # Initialize results
    results = {
        'num_examples': len(filtered_examples),
        'num_filtered': len(examples) - len(filtered_examples),
        'vanilla': {'correct_count': 0, 'per_example': []},
        'cot': {'correct_count': 0, 'per_example': []},
        'patchscope': {
            'accuracy_by_layer': {},
            'per_example': []
        }
    }

    print(f"\nPhase 1: Running vanilla and patchscope on {len(filtered_examples)} examples...")

    # Process each example
    for example_idx, example in enumerate(tqdm(filtered_examples, desc="Vanilla+Patchscope")):
        example_results = {
            'id': example.get('id', f'example_{example_idx}'),
            'question': example['combined_question'],
            'target_answer': example['hop2_answer'],
            'vanilla': {},
            'cot': {},
            'patchscope_by_layer': {}
        }

        # Method 1: Vanilla
        vanilla_prompt = build_vanilla_multihop_prompt(example['combined_question'])
        vanilla_tokens, _ = vanilla_generate(
            model, vanilla_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        vanilla_correct = answer_appears_in_generation(vanilla_tokens, example['hop2_answer'])

        example_results['vanilla'] = {
            'generated': "".join(vanilla_tokens),
            'correct': vanilla_correct
        }
        if vanilla_correct:
            results['vanilla']['correct_count'] += 1

        # Method 2: Patchscope (sweep layers)
        hop1_prompt = build_hop1_prompt(example['hop1_prompt'], placeholder)
        hop2_prompt = build_hop2_prompt(example['hop2_prompt_prefix'], placeholder)

        # Find positions - extract from the last token of the hop1 query phrase
        hop1_position = find_substring_token_position(model, hop1_prompt, example['hop1_prompt'], return_last=True)
        if hop1_position is None:
            hop1_position = len(model.to_tokens(hop1_prompt, prepend_bos=True)[0]) - 1

        hop2_placeholder_pos = find_token_position(model, hop2_prompt, placeholder)
        if hop2_placeholder_pos is None:
            print(f"Warning: Could not find placeholder in hop2 prompt for example {example_idx}")
            continue

        for layer in layers:
            try:
                # Extract bridge entity representation from hop1
                with torch.no_grad():
                    bridge_vec = extract_representation(
                        model,
                        hop1_prompt,
                        layer=layer,
                        position=hop1_position
                    )

                # Patch into hop2 and generate
                with torch.no_grad():
                    patch_tokens, _ = generate_with_patch(
                        model,
                        hop2_prompt,
                        target_position=hop2_placeholder_pos,
                        source_vec=bridge_vec,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )

                patch_correct = answer_appears_in_generation(patch_tokens, example['hop2_answer'])

                example_results['patchscope_by_layer'][layer] = {
                    'generated': "".join(patch_tokens),
                    'correct': patch_correct
                }

            except Exception as e:
                print(f"Error at layer {layer}, example {example_idx}: {e}")
                example_results['patchscope_by_layer'][layer] = {
                    'generated': "",
                    'correct': False,
                    'error': str(e)
                }

        results['patchscope']['per_example'].append(example_results)
        results['vanilla']['per_example'].append(example_results)
        results['cot']['per_example'].append(example_results)

    # Compute vanilla accuracy
    results['vanilla']['accuracy'] = results['vanilla']['correct_count'] / len(filtered_examples)

    # Compute patchscope accuracy by layer
    for layer in layers:
        layer_correct = sum(
            1 for ex in results['patchscope']['per_example']
            if ex['patchscope_by_layer'].get(layer, {}).get('correct', False)
        )
        results['patchscope']['accuracy_by_layer'][layer] = layer_correct / len(filtered_examples)

    # Find best layer
    if results['patchscope']['accuracy_by_layer']:
        best_layer = max(results['patchscope']['accuracy_by_layer'],
                        key=results['patchscope']['accuracy_by_layer'].get)
        results['patchscope']['best_layer'] = best_layer
        results['patchscope']['best_accuracy'] = results['patchscope']['accuracy_by_layer'][best_layer]

    return results, filtered_examples


def run_cot_phase(
    cot_model,
    results,
    filtered_examples,
    max_new_tokens=20,
    temperature=0.0,
    use_instruct_format=False,
):
    """
    Run CoT experiment (Phase 2).

    Args:
        cot_model: HookedTransformer model (instruct model for CoT)
        results: Results dict from Phase 1
        filtered_examples: Examples used in Phase 1
        max_new_tokens: Tokens to generate
        temperature: Generation temperature (0 = greedy)
        use_instruct_format: Whether to use instruct format for CoT prompts

    Returns:
        results: Updated results dict with CoT results
    """
    print(f"\nPhase 2: Running CoT on {len(filtered_examples)} examples...")

    # Process each example for CoT
    for example_idx, example in enumerate(tqdm(filtered_examples, desc="CoT")):
        cot_prompt = build_cot_multihop_prompt(example['combined_question'], use_instruct_format=use_instruct_format)
        cot_tokens, _ = vanilla_generate(
            cot_model, cot_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        cot_correct = answer_appears_in_generation(cot_tokens, example['hop2_answer'])

        # Update the example results
        results['cot']['per_example'][example_idx]['cot'] = {
            'generated': "".join(cot_tokens),
            'correct': cot_correct
        }
        if cot_correct:
            results['cot']['correct_count'] += 1

    # Compute CoT accuracy
    results['cot']['accuracy'] = results['cot']['correct_count'] / len(filtered_examples)

    return results


def run_multihop_experiment(
    model,
    examples,
    layers,
    placeholder=" x",
    max_new_tokens=20,
    temperature=0.0,
    filter_solvable=False,
    check_n_tokens=20,
    cot_model=None,
    use_instruct_format=False,
):
    """
    Run multi-hop reasoning experiment comparing three methods.
    
    Note: If cot_model is the same as model, runs everything in one pass.
    Otherwise, caller should use run_vanilla_and_patchscope + run_cot_phase separately
    to manage memory.

    For each example:
    1. Optionally filter: check if both hops are individually solvable
    2. Vanilla: Generate from combined question directly
    3. CoT: Generate with "Let's think step by step" prefix
    4. Patchscope: Extract hop1 bridge entity, patch into hop2

    Args:
        model: HookedTransformer model (base model for vanilla and patchscope)
        examples: List of multi-hop example dicts
        layers: List of layer indices to test for patchscope
        placeholder: Single-token placeholder
        max_new_tokens: Tokens to generate for each method
        temperature: Generation temperature (0 = greedy)
        filter_solvable: Whether to pre-filter examples
        check_n_tokens: Tokens to generate for solvability checks
        cot_model: Optional separate model for CoT (if None, uses model)
        use_instruct_format: Whether to use instruct format for CoT prompts

    Returns:
        results: Dict with structure:
            {
                'num_examples': int,
                'num_filtered': int,
                'vanilla': {
                    'accuracy': float,
                    'correct_count': int,
                    'per_example': [...]
                },
                'cot': {
                    'accuracy': float,
                    'correct_count': int,
                    'per_example': [...]
                },
                'patchscope': {
                    'accuracy_by_layer': {layer: float, ...},
                    'best_layer': int,
                    'best_accuracy': float,
                    'per_example': [...]
                }
            }
    """
    # Run phase 1: vanilla + patchscope
    results, filtered_examples = run_vanilla_and_patchscope(
        model, examples, layers,
        placeholder=placeholder,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        filter_solvable=filter_solvable,
        check_n_tokens=check_n_tokens
    )

    # Run phase 2: CoT
    cot_model_to_use = cot_model if cot_model is not None else model
    results = run_cot_phase(
        cot_model_to_use, results, filtered_examples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_instruct_format=use_instruct_format
    )

    return results
