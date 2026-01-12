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
