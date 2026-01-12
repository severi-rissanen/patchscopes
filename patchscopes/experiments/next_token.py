"""Next-token prediction experiment across layers (Fig. 2 reproduction)."""
import torch
import numpy as np
from tqdm import tqdm

from patchscopes.baselines import logit_lens
from patchscopes.metrics import precision_at_1, surprisal
from patchscopes.prompts import build_identity_prompt
from patchscopes.positions import find_token_position, verify_single_token
from patchscopes.patch import extract_representation, run_with_patch


def run_next_token_experiment(
    model,
    samples,
    layers,
    placeholder=" x",
    num_demos=5,
):
    """
    Run next-token prediction experiment across layers.

    For each layer and sample:
    1. Compute ground truth logits (full model forward pass)
    2. Extract hidden state at layer ℓ
    3. Decode using two methods:
       a) Logit Lens baseline
       b) Token Identity Patchscope
    4. Compute Precision@1 and Surprisal for each method

    Args:
        model: HookedTransformer model
        samples: List of (text, position) tuples
        layers: List of layer indices to evaluate
        placeholder: Single-token placeholder for identity prompt
        num_demos: Number of demonstrations in identity prompt

    Returns:
        results: Dict with structure:
            {
                'layers': [0, 5, 10, ...],
                'logit_lens': {
                    'precision_at_1': [0.2, 0.4, ...],  # averaged across samples
                    'surprisal': [5.2, 3.1, ...],
                },
                'patchscope': {
                    'precision_at_1': [...],
                    'surprisal': [...],
                }
            }
    """
    # Verify placeholder is single token
    if not verify_single_token(model, placeholder):
        # Try alternative placeholders
        for alt in [" X", " z", " #"]:
            if verify_single_token(model, alt):
                placeholder = alt
                print(f"Using placeholder: '{placeholder}'")
                break
        else:
            raise ValueError(f"Could not find single-token placeholder. Tried: {placeholder}, ' X', ' z', ' #'")

    # Build identity prompt once (reuse for all samples)
    identity_prompt = build_identity_prompt(placeholder, num_demos)
    placeholder_pos = find_token_position(model, identity_prompt, placeholder)

    if placeholder_pos is None:
        raise ValueError(f"Could not find placeholder '{placeholder}' in identity prompt")

    print(f"Identity prompt: '{identity_prompt}'")
    print(f"Placeholder position: {placeholder_pos}")
    print(f"Evaluating on {len(samples)} samples across {len(layers)} layers")
    print()

    # Initialize result storage
    results = {
        'layers': layers,
        'logit_lens': {'precision_at_1': [], 'surprisal': []},
        'patchscope': {'precision_at_1': [], 'surprisal': []},
    }

    # For each layer
    for layer in layers:
        print(f"Processing layer {layer}...")

        ll_p1_scores = []
        ll_surp_scores = []
        ps_p1_scores = []
        ps_surp_scores = []

        # For each sample
        for text, position in tqdm(samples, desc=f"Layer {layer}", leave=False):
            # 1. Get ground truth logits
            tokens = model.to_tokens(text, prepend_bos=True)

            with torch.no_grad():
                logits_true_full = model(tokens)
            logits_true = logits_true_full[0, position, :]  # [vocab_size]

            # 2. Extract hidden state at layer ℓ, position
            with torch.no_grad():
                hidden_state = extract_representation(model, text, layer, position)

            # 3a. Logit Lens prediction
            with torch.no_grad():
                logits_ll = logit_lens(model, hidden_state)
            ll_p1_scores.append(precision_at_1(logits_ll, logits_true))
            ll_surp_scores.append(surprisal(logits_ll, logits_true))

            # 3b. Patchscope prediction
            with torch.no_grad():
                logits_ps_full = run_with_patch(
                    model,
                    identity_prompt,
                    placeholder_pos,
                    hidden_state
                )
            logits_ps = logits_ps_full[0, placeholder_pos, :]  # [vocab_size]
            ps_p1_scores.append(precision_at_1(logits_ps, logits_true))
            ps_surp_scores.append(surprisal(logits_ps, logits_true))

        # Aggregate across samples (mean)
        results['logit_lens']['precision_at_1'].append(np.mean(ll_p1_scores))
        results['logit_lens']['surprisal'].append(np.mean(ll_surp_scores))
        results['patchscope']['precision_at_1'].append(np.mean(ps_p1_scores))
        results['patchscope']['surprisal'].append(np.mean(ps_surp_scores))

        # Print summary for this layer
        print(f"  Logit Lens    - P@1: {results['logit_lens']['precision_at_1'][-1]:.3f}, "
              f"Surprisal: {results['logit_lens']['surprisal'][-1]:.3f}")
        print(f"  Patchscope    - P@1: {results['patchscope']['precision_at_1'][-1]:.3f}, "
              f"Surprisal: {results['patchscope']['surprisal'][-1]:.3f}")
        print()

    return results
