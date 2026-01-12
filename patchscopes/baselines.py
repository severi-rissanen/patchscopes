"""Baseline methods for decoding intermediate layer representations."""
import torch


def logit_lens(model, hidden_state):
    """
    Apply Logit Lens: map hidden state directly to vocabulary logits.

    For Llama-style models, this involves:
    1. Apply final layer norm (ln_final)
    2. Apply unembedding matrix (unembed)

    This is a "direct projection" baseline that treats the intermediate
    layer representation as if it were already in "vocab space", without
    any learned transformation.

    Args:
        model: HookedTransformer model
        hidden_state: [d_model] hidden representation from intermediate layer

    Returns:
        logits: [vocab_size] predicted logits
    """
    # Apply final layer norm
    normed = model.ln_final(hidden_state)

    # Apply unembedding matrix
    logits = model.unembed(normed)

    return logits
