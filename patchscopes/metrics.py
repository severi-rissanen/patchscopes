"""Evaluation metrics for comparing predicted vs ground-truth distributions."""
import torch


def precision_at_1(pred_logits, true_logits):
    """
    Compute Precision@1 metric.

    Returns 1.0 if the predicted token (argmax of pred_logits) matches
    the true token (argmax of true_logits), otherwise 0.0.

    Args:
        pred_logits: [vocab_size] predicted logits
        true_logits: [vocab_size] ground truth logits

    Returns:
        1.0 if argmax matches, 0.0 otherwise
    """
    pred_token = torch.argmax(pred_logits)
    true_token = torch.argmax(true_logits)
    return 1.0 if pred_token == true_token else 0.0


def surprisal(pred_logits, true_logits):
    """
    Compute surprisal of predicted token under true distribution.

    Surprisal = -log p_true(argmax(p_pred))

    This measures how "surprising" the predicted token is under the
    true distribution. Lower surprisal means the predicted token has
    high probability under the true distribution.

    Args:
        pred_logits: [vocab_size] predicted logits
        true_logits: [vocab_size] ground truth logits

    Returns:
        Surprisal value (float)
    """
    pred_token = torch.argmax(pred_logits)
    true_log_probs = torch.log_softmax(true_logits.to(torch.float32), dim=-1)

    return -true_log_probs[pred_token].item()


def rouge_l(predicted_text, reference_text):
    """
    Compute ROUGE-L score between predicted and reference text.

    ROUGE-L measures the longest common subsequence (LCS) between texts,
    which captures sentence-level structure similarity.

    Args:
        predicted_text: Generated text string
        reference_text: Ground truth reference text string

    Returns:
        ROUGE-L F1 score (float between 0 and 1)
        Returns 0.0 for edge cases (empty strings, None values)
    """
    from rouge_score import rouge_scorer

    # Handle edge cases
    if not predicted_text or not reference_text:
        return 0.0

    if not isinstance(predicted_text, str) or not isinstance(reference_text, str):
        return 0.0

    # Strip whitespace
    predicted_text = predicted_text.strip()
    reference_text = reference_text.strip()

    if not predicted_text or not reference_text:
        return 0.0

    # Initialize ROUGE scorer with ROUGE-L metric
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Compute scores
    scores = scorer.score(reference_text, predicted_text)

    # Return F1 score (harmonic mean of precision and recall)
    return scores['rougeL'].fmeasure


def answer_appears_in_generation(generated_tokens, target_answer, case_insensitive=True):
    """
    Check if target answer appears in generated token sequence.

    This is the success metric for multi-hop reasoning experiments.
    The answer is considered "found" if it appears as a substring anywhere
    in the generated text.

    Args:
        generated_tokens: List of generated token strings
        target_answer: Target answer string to find
        case_insensitive: Whether to do case-insensitive matching (default: True)

    Returns:
        True if answer appears, False otherwise
    """
    # Join all generated tokens into a single string
    generated_text = "".join(generated_tokens)

    # Perform substring matching
    if case_insensitive:
        return target_answer.lower() in generated_text.lower()
    else:
        return target_answer in generated_text
