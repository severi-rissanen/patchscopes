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


def load_entities(file_path):
    """
    Load entity data from JSON or CSV file.

    Expected JSON format:
        [
            {
                "entity": "Diana, Princess of Wales",
                "reference": "Member of the British royal family"
            },
            ...
        ]

    Expected CSV format (with header):
        entity,reference
        "Diana, Princess of Wales","Member of the British royal family"
        ...

    Args:
        file_path: Path to JSON or CSV file

    Returns:
        List of dicts with 'entity' and 'reference' keys

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    import json
    import csv
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Entity data file not found: {file_path}")

    # Determine format from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)

        # Validate format
        if not isinstance(entities, list):
            raise ValueError("JSON must contain a list of entity objects")

        for i, item in enumerate(entities):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dict")
            if 'entity' not in item or 'reference' not in item:
                raise ValueError(f"Item {i} missing 'entity' or 'reference' field")
            if not isinstance(item['entity'], str) or not isinstance(item['reference'], str):
                raise ValueError(f"Item {i} has non-string entity or reference")

    elif ext == '.csv':
        entities = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate header
            if 'entity' not in reader.fieldnames or 'reference' not in reader.fieldnames:
                raise ValueError("CSV must have 'entity' and 'reference' columns")

            for i, row in enumerate(reader):
                if not row['entity'].strip() or not row['reference'].strip():
                    raise ValueError(f"Row {i+2} has empty entity or reference")

                entities.append({
                    'entity': row['entity'].strip(),
                    'reference': row['reference'].strip()
                })

    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .json or .csv")

    if not entities:
        raise ValueError("Entity file is empty")

    return entities


def validate_entities(entities):
    """
    Validate entity data structure and content.

    Args:
        entities: List of entity dicts

    Returns:
        Number of valid entities

    Prints warnings for any issues found.
    """
    valid_count = 0

    for i, item in enumerate(entities):
        entity_name = item.get('entity', '')
        reference = item.get('reference', '')

        # Check for very short entities
        if len(entity_name) < 2:
            print(f"Warning: Entity {i} has very short name: '{entity_name}'")
            continue

        # Check for very short references
        if len(reference) < 5:
            print(f"Warning: Entity {i} has very short reference: '{reference}'")
            continue

        # Check for very long references (might indicate formatting issues)
        if len(reference) > 500:
            print(f"Warning: Entity {i} has very long reference ({len(reference)} chars)")

        valid_count += 1

    return valid_count
