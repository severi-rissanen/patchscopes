"""Token position utilities for Patchscopes."""
import torch


def find_token_position(model, text, target_token_str):
    """
    Find the position of a target token string in the tokenized text.

    Args:
        model: HookedTransformer model (for tokenizer access)
        text: Input text string
        target_token_str: Token string to find (e.g., "CEO", " x")

    Returns:
        Position index (int) or None if not found
    """
    tokens = model.to_tokens(text, prepend_bos=True)

    # Tokenize the target string to see what token ID we're looking for
    target_tokens = model.to_tokens(target_token_str, prepend_bos=False)

    # Find the position of the target token
    for i in range(tokens.shape[1]):
        if tokens[0, i] == target_tokens[0, 0]:
            return i

    return None


def find_substring_token_position(model, text, substring):
    """
    Find the position of the last token that represents a substring.

    Useful when the substring might be split across multiple tokens.

    Args:
        model: HookedTransformer model
        text: Input text string
        substring: Substring to find

    Returns:
        Position index of the last token of the substring
    """
    tokens = model.to_tokens(text, prepend_bos=True)

    # Decode each prefix to find where the substring ends
    for i in range(1, tokens.shape[1]):
        decoded = model.to_string(tokens[0, :i+1])
        if substring in decoded:
            # Check if this is the last token of the substring
            if i == tokens.shape[1] - 1 or substring not in model.to_string(tokens[0, :i+2]).replace(model.to_string(tokens[0, :i+1]), ""):
                return i

    return None


def verify_single_token(model, token_str):
    """
    Verify that a string tokenizes to exactly one token.

    Args:
        model: HookedTransformer model
        token_str: String to check

    Returns:
        True if single token, False otherwise
    """
    tokens = model.to_tokens(token_str, prepend_bos=False)
    return tokens.shape[1] == 1
