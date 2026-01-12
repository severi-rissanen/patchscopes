"""Data loading and sampling utilities for experiments."""
import random


def load_wikitext2(split='test'):
    """
    Load WikiText-2 dataset.

    Args:
        split: 'train', 'validation', or 'test'
        num_samples: Maximum number of samples to use

    Returns:
        List of text strings (empty lines filtered out)
    """
    from datasets import load_dataset

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    # Filter out empty lines (WikiText has many)
    texts = [text for text in dataset['text'] if text.strip()]

    # Take first num_samples
    return texts


def sample_positions(model, texts, positions_per_text=1, min_length=20, max_length=100):
    """
    Sample valid positions from texts for next-token prediction.

    For each text, we:
    1. Tokenize it
    2. Skip if too short or too long
    3. Sample random positions that have valid next tokens

    Args:
        model: HookedTransformer model (for tokenization)
        texts: List of text strings
        positions_per_text: How many positions to sample per text
        min_length: Minimum token length to consider
        max_length: Maximum token length to use (for memory efficiency)

    Returns:
        List of (text, position) tuples where position is valid for prediction
    """
    samples = []

    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=True)
        seq_len = tokens.shape[1]

        # Skip if too short or too long
        if seq_len < min_length or seq_len > max_length:
            continue

        # Sample random positions (not last position, as there's no next token)
        # Also avoid BOS position (0)
        valid_positions = list(range(5, seq_len - 1))

        if len(valid_positions) < positions_per_text:
            continue

        sampled_pos = random.sample(valid_positions, positions_per_text)

        for pos in sampled_pos:
            samples.append((text, pos))

    return samples
