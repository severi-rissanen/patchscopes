# Patchscopes

Minimal reproduction of the some of the main results in the [Patchscopes paper](https://arxiv.org/abs/2401.06102) using TransformerLens and Llama-3-8B.

Patchscopes is a framework for inspecting hidden representations in language models by "patching" them into different computational contexts and observing the outputs.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformer_lens` - For HookedTransformer models
- `torch` - Deep learning framework
- `datasets` - For WikiText-2
- `matplotlib` - Visualization
- `rouge-score` - ROUGE-L metric

### Model Access

You'll need HuggingFace access to Llama-3 weights:
```bash
huggingface-cli login
```

## Repository Structure

```
patchscopes/
├── patchscopes/          # Core library
│   ├── model.py          # Model loading utilities
│   ├── patch.py          # Patchscope extraction & patching
│   ├── prompts.py        # Prompt builders
│   ├── positions.py      # Token position utilities
│   ├── metrics.py        # Evaluation metrics
│   ├── baselines.py      # Baseline methods (Logit Lens)
│   ├── data.py           # Dataset loading
│   └── experiments/      # Experiment implementations
│       ├── next_token.py
│       ├── entity.py
│       └── multi_hop.py
├── scripts/              # Runnable experiments
├── data/                 # Example data (entities, multi-hop questions)
```

## Experiments

Run experiments in this order to replicate key paper results:

### 1. Figure 1 Demo: Identity Patchscope

Interactive demonstration of the basic Patchscope mechanism on a single token.

```bash
python scripts/demo_fig1.py
```

### 2. Figure 2: Next-Token Prediction Across Layers

Compares Logit Lens vs Token Identity Patchscope for decoding next-token predictions.

```bash
python scripts/run_fig2.py --samples 500
```

**Output:** Plot showing Precision@1 and Surprisal across layers. An example of this is shown in fig2_results.png.

### 3. Section 4.3: Entity Resolution

Tests when models resolve entity meanings across layers using ROUGE-L.

**Interactive (single entity):**
```bash
python scripts/run_entity_interactive.py "Diana, Princess of Wales" --layer-range 0-15
```

**Full experiment (multiple entities):**
```bash
python scripts/run_entity.py data/entities.json --layers 0,1,2,3,4,5,6,7,8,9
```

**Output:** Plot showing ROUGE-L scores across layers, JSON with detailed results. An example output is shown in entity_results.png and entity_results.json. 

### 4. Multi-Hop Reasoning

Tests whether Patchscopes can fix composition failures in 2-hop reasoning by patching entity representations between queries focusing on individual hops. This does not seem to work as well as in the original paper so far, potentially due to the deficiencies in the toy dataset in data/multihop_examples.json, or how the models are prompted. 

**Interactive (single question):**
```bash
python scripts/run_multi_hop_interactive.py --example-id 0 --layer-range 0-9
```

**Full experiment (20 questions):**
```bash
python scripts/run_multi_hop.py data/multihop_examples.json \
  --layers 0,1,2,3,4,5,6,7,8,9 \
  --output-plot multihop_results.png \
  --output-json multihop_results.json
```

**Output:** Plot comparing vanilla/CoT/Patchscope methods, JSON with per-example results. 

## Citation for the original authors

```bibtex
@inproceedings{ghandeharioun2024patchscopes,
  title={Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models},
  author={Ghandeharioun, Asma and Caciularu, Avi and Pearce, Adam and Dixon, Lucas and Geva, Mor},
  booktitle={International Conference on Machine Learning},
  pages={15466--15490},
  year={2024},
  organization={PMLR}
}
```
