"""Core patchscoping functionality: extract and patch representations."""
import torch


def extract_representation(model, text, layer, position, hook_name=None):
    """
    Extract a representation from a specific layer and position.

    Args:
        model: HookedTransformer model
        text: Source text string
        layer: Layer number to extract from
        position: Token position to extract
        hook_name: Hook name (default: f"blocks.{layer}.hook_resid_post")

    Returns:
        Extracted representation tensor [d_model]
    """
    if hook_name is None:
        hook_name = f"blocks.{layer}.hook_resid_post"

    # Tokenize
    tokens = model.to_tokens(text, prepend_bos=True)

    # Run with cache, filtering to only the hook we need
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

    # Extract the representation at the specified position
    representation = cache[hook_name][0, position, :]  # [d_model]

    return representation


def make_patch_hook(source_vec, target_position):
    """
    Create a hook function that patches a source vector into a target position.

    Args:
        source_vec: Source representation tensor [d_model]
        target_position: Position to patch into

    Returns:
        Hook function compatible with TransformerLens
    """
    def hook_fn(resid, hook):
        """
        Hook function that overwrites the residual stream at target_position.

        Args:
            resid: Residual stream tensor [batch, pos, d_model]
            hook: Hook object (unused)

        Returns:
            Modified residual stream
        """
        resid[:, target_position, :] = source_vec
        return resid

    return hook_fn


def run_with_patch(model, target_text, layer, target_position, source_vec, hook_name=None):
    """
    Run the model on target text with a patched representation.

    Args:
        model: HookedTransformer model
        target_text: Target prompt string
        layer: Layer to patch into
        target_position: Position to patch into
        source_vec: Source representation to patch [d_model]
        hook_name: Hook name (default: f"blocks.{layer}.hook_resid_post")

    Returns:
        logits: Output logits [batch, pos, vocab_size]
    """
    if hook_name is None:
        hook_name = f"blocks.{layer}.hook_resid_post"

    # Tokenize target
    tokens = model.to_tokens(target_text, prepend_bos=True)

    # Create the patch hook
    hook_fn = make_patch_hook(source_vec, target_position)

    # Run with the hook
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, hook_fn)]
    )

    return logits


def get_top_tokens(model, logits, position, k=10):
    """
    Get the top-k predicted tokens and their probabilities.

    Args:
        model: HookedTransformer model (for decoding)
        logits: Model logits [batch, pos, vocab_size]
        position: Position to get predictions for
        k: Number of top tokens to return

    Returns:
        List of (token_string, probability) tuples
    """
    # Get logits at position and convert to probabilities
    logits_at_pos = logits[0, position, :]  # [vocab_size]
    probs = torch.softmax(logits_at_pos, dim=-1)

    # Get top-k
    top_probs, top_indices = torch.topk(probs, k)

    # Decode tokens
    results = []
    for prob, idx in zip(top_probs, top_indices):
        token_str = model.to_string(idx)
        results.append((token_str, prob.item()))

    return results
