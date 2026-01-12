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
    Pattern: "descriptive phrase -> Entity Name ; ... ; <hop1_text> ->"

    This encourages the model to "name" the bridge entity implied by hop1_text.

    Args:
        hop1_text: Descriptive phrase for hop 1 (e.g., "the company that created Visual Basic")
        placeholder: Single-token placeholder (not used in this prompt, but kept for consistency)

    Returns:
        Prompt string for hop 1
    """
    # Diverse few-shot examples showing entity naming pattern
    # Covers: tech companies, people, places, organizations, products
    demos = [
        # Tech companies - varied phrasings
        "the search engine giant founded by Larry Page -> Google",
        "the company that makes the iPhone -> Apple",
        "the creator of Windows and Office -> Microsoft",
        "the e-commerce company founded by Jeff Bezos -> Amazon",
        "the social media platform founded by Mark Zuckerberg -> Facebook",
        "the electric car company led by Elon Musk -> Tesla",
        # People
        "the physicist who developed the theory of relativity -> Einstein",
        "the founder of Microsoft -> Bill Gates",
        "the current CEO of Apple -> Tim Cook",
        # Places
        "the capital of France -> Paris",
        "the largest city in Japan -> Tokyo",
        # Products/Brands
        "the programming language created by Guido van Rossum -> Python",
        "the AI assistant made by OpenAI -> ChatGPT",
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


def build_cot_multihop_prompt(combined_question, use_instruct_format=False):
    """
    Build chain-of-thought prompt for 2-hop question.

    Adds "Let's think step by step." to encourage reasoning.
    When use_instruct_format=True, formats for Llama 3 Instruct chat template.

    Args:
        combined_question: Full 2-hop question
        use_instruct_format: Whether to use Llama 3 Instruct chat format

    Returns:
        Prompt with CoT trigger
    """
    if use_instruct_format:
        # Llama 3 Instruct chat template format
        system_msg = "You are a helpful assistant. Answer questions by thinking step by step."
        user_msg = f"Question: {combined_question}\n\nLet's think step by step to find the answer."
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt
    else:
        return f"Let's think step by step. {combined_question} is"
