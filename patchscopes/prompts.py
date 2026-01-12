"""Prompt builders for Patchscopes experiments."""


def build_identity_prompt(placeholder=" x", num_demos=5):
    """
    Build an identity-style target prompt for token decoding.

    The prompt follows the pattern:
    "token1 -> token1 ; token2 -> token2 ; ... ; placeholder"

    This encourages the model to decode what a patched representation means
    by predicting the next token(s) after the placeholder.

    Args:
        placeholder: Single-token string to use as placeholder (default " x")
        num_demos: Number of demonstration pairs to include

    Returns:
        Identity prompt string
    """
    # Use simple single tokens for demonstrations
    demo_tokens = ["cat", "dog", "hello", "world", "apple"][:num_demos]

    # Build demonstrations: "token -> token"
    demos = [f"{tok} -> {tok}" for tok in demo_tokens]

    # Join with " ; " and add placeholder at the end
    prompt = " ; ".join(demos) + f" ; {placeholder} ->"

    return prompt


def build_entity_description_prompt(entities_with_descriptions, placeholder=" x"):
    """
    Build an entity-description style target prompt.

    The prompt follows the pattern:
    "entity1: description1, entity2: description2, ..., placeholder:"

    Used for entity resolution experiments (§4.3).

    Args:
        entities_with_descriptions: List of (entity, description) tuples
        placeholder: Single-token string to use as placeholder

    Returns:
        Entity description prompt string
    """
    # Build demonstrations: "entity: description"
    demos = [f"{entity}: {desc}" for entity, desc in entities_with_descriptions]

    # Join with ", " and add placeholder at the end
    prompt = ", ".join(demos) + f", {placeholder}:"

    return prompt
