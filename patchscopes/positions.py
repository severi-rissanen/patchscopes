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


def find_substring_token_position(model, text, substring, return_last=True):
    """
    Find the token position corresponding to a substring in the text.

    Useful when the substring might be split across multiple tokens.

    Args:
        model: HookedTransformer model
        text: Input text string
        substring: Substring to find
        return_last: If True, return position of last token of substring.
                     If False, return position of first token.

    Returns:
        Position index of the first or last token of the substring, or None if not found
        (position accounts for prepend_bos=True, consistent with model usage)
    """
    tokens = model.to_tokens(text, prepend_bos=True)
    
    # Find character position of substring in text
    char_start = text.find(substring)
    if char_start == -1:
        return None
    char_end = char_start + len(substring)
    
    # Map character positions to token positions by decoding prefixes
    # Skip BOS token (index 0) when decoding to match original text character positions
    first_token_pos = None
    last_token_pos = None
    
    for i in range(1, tokens.shape[1] + 1):
        # Decode starting from token 1 (skip BOS) to match original text
        decoded = model.to_string(tokens[0, 1:i+1])
        decoded_len = len(decoded)
        
        # Find first token that covers the start of substring
        # Position is i (not i-1) because we account for BOS at position 0
        if first_token_pos is None and decoded_len > char_start:
            first_token_pos = i
        
        # Find last token that covers the end of substring
        if decoded_len >= char_end:
            last_token_pos = i
            break
    
    if return_last:
        return last_token_pos
    else:
        return first_token_pos


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
