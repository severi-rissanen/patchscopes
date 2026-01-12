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


def run_with_patch(model, target_text, target_position, source_vec, hook_name=None):
    """
    Run the model on target text with a patched representation at layer 0.

    Args:
        model: HookedTransformer model
        target_text: Target prompt string
        target_position: Position to patch into
        source_vec: Source representation to patch [d_model]
        hook_name: Hook name (default: "blocks.0.hook_resid_pre")

    Returns:
        logits: Output logits [batch, pos, vocab_size]
    """
    if hook_name is None:
        hook_name = "blocks.0.hook_resid_pre"

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


def generate_with_patch(
    model,
    target_text,
    target_position,
    source_vec,
    max_new_tokens=5,
    hook_name=None,
    temperature=1.0,
):
    """
    Patch a representation at layer 0 and generate tokens autoregressively.

    The patch is applied on every forward pass at layer 0 and the target position,
    ensuring the patched representation influences all generated tokens.

    Args:
        model: HookedTransformer model
        target_text: Target prompt string
        target_position: Position to patch into
        source_vec: Source representation to patch [d_model]
        max_new_tokens: Number of tokens to generate
        hook_name: Hook name (default: "blocks.0.hook_resid_pre")
        temperature: Sampling temperature (0 = greedy argmax)

    Returns:
        generated_tokens: List of generated token strings
        generated_ids: Tensor of generated token ids
    """
    if hook_name is None:
        hook_name = "blocks.0.hook_resid_pre"

    # Tokenize target
    tokens = model.to_tokens(target_text, prepend_bos=True)

    # Create the patch hook (reused for all forward passes)
    hook_fn = make_patch_hook(source_vec, target_position)

    generated_ids = []

    # Generate tokens autoregressively, applying patch on each forward pass
    for _ in range(max_new_tokens):
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)]
        )

        # Get next token prediction from last position
        next_token_logits = logits[0, -1, :]
        if temperature <= 0:
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        generated_ids.append(next_token_id.item())
        tokens = torch.cat([tokens, next_token_id], dim=1)

    # Decode generated tokens
    generated_tokens = [model.to_string(tid) for tid in generated_ids]

    return generated_tokens, torch.tensor(generated_ids)


def vanilla_generate(model, prompt, max_new_tokens=20, temperature=0.0):
    """
    Generate tokens without any patching (vanilla baseline).

    This is used for vanilla and CoT baselines in multi-hop experiments.

    Args:
        model: HookedTransformer model
        prompt: Input prompt string
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (0 = greedy argmax)

    Returns:
        generated_tokens: List of generated token strings
        generated_ids: Tensor of generated token ids
    """
    # Tokenize
    tokens = model.to_tokens(prompt, prepend_bos=True)
    generated_ids = []

    # Generate tokens autoregressively
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)

        # Get next token prediction from last position
        next_token_logits = logits[0, -1, :]
        if temperature <= 0:
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        generated_ids.append(next_token_id.item())
        tokens = torch.cat([tokens, next_token_id], dim=1)

    # Decode generated tokens
    generated_tokens = [model.to_string(tid) for tid in generated_ids]

    return generated_tokens, torch.tensor(generated_ids)
