"""Model loading utilities for TransformerLens."""
import torch
from transformer_lens import HookedTransformer


def load_model(model_name="meta-llama/Meta-Llama-3-8B", device="cuda", dtype=torch.float16, seed=42):
    """
    Load a HookedTransformer model.

    Args:
        model_name: HuggingFace model name (must be supported by TransformerLens)
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Model dtype (torch.float16 or torch.float32)
        seed: Random seed for reproducibility

    Returns:
        HookedTransformer model
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )

    return model
