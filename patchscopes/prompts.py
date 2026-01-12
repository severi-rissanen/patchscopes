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
    prompt = " ; ".join(demos) + f" ; {placeholder}"

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


def build_hop1_prompt(hop1_text, placeholder=" x"):
    """
    Build hop 1 prompt for extracting bridge entity representation.

    Uses identity-style prompt to decode bridge entity from a descriptive phrase.
    Pattern: "the search engine company -> Google ; the iPhone maker -> Apple ; <hop1_text> ->"

    This encourages the model to "name" the bridge entity implied by hop1_text.

    Args:
        hop1_text: Descriptive phrase for hop 1 (e.g., "the company that created Visual Basic")
        placeholder: Single-token placeholder (not used in this prompt, but kept for consistency)

    Returns:
        Prompt string for hop 1
    """
    # Few-shot examples showing entity naming pattern
    demos = [
        "the search engine company -> Google",
        "the iPhone maker -> Apple",
        "the Windows creator -> Microsoft"
    ]

    prompt = " ; ".join(demos) + f" ; {hop1_text} ->"

    return prompt


def build_hop2_prompt(hop2_prefix, placeholder=" x"):
    """
    Build hop 2 prompt with placeholder for bridge entity.

    Pattern: "The current CEO of [placeholder] is"

    The placeholder will be patched with the bridge entity representation extracted from hop 1.

    Args:
        hop2_prefix: Relation phrase for hop 2 (e.g., "The current CEO of")
        placeholder: Single-token placeholder where bridge entity will be patched

    Returns:
        Prompt string for hop 2 with placeholder
    """
    # Construct prompt with placeholder standing in for bridge entity
    # Add "is" at the end to encourage answer completion
    prompt = f"{hop2_prefix} {placeholder} is"

    return prompt


def build_vanilla_multihop_prompt(combined_question):
    """
    Build vanilla prompt for direct 2-hop question answering.

    This is the baseline that doesn't use any prompting tricks.

    Args:
        combined_question: Full 2-hop question (e.g., "The current CEO of the company that created Visual Basic")

    Returns:
        Simple prompt string
    """
    return f"{combined_question} is"


def build_cot_multihop_prompt(combined_question):
    """
    Build chain-of-thought prompt for 2-hop question.

    Adds "Let's think step by step." as a prefix to encourage reasoning.

    Args:
        combined_question: Full 2-hop question

    Returns:
        Prompt with CoT trigger
    """
    return f"Let's think step by step. {combined_question} is"
