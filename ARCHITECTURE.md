# Architecture

This project implements a three-stage pipeline: fine-tune a model on the TOFU dataset, unlearn a subset of training data, then probabilistically evaluate how much information was retained.

---

## Module Overview

```
probabilistic-unlearning/
├── finetuning/     – fine-tune a base model on the full TOFU dataset
├── unlearning/     – apply an unlearning method to a fine-tuned checkpoint
├── evaluation/     – probabilistically evaluate leakage of an unlearned model
├── results/        – output directory (auto-created by evaluation scripts)
└── result/         – legacy output directory (kept for reference)
```

Each of `finetuning/`, `unlearning/`, and `evaluation/` is self-contained with its own entry point (`main.py` or standalone scripts), configs, and dependencies.

---

## Data Flow

```
[Base LLM (e.g. Phi-1.5)]
        │
        │  finetuning/main.py
        │  Config: finetuning/configs/phi.yaml
        ▼
[Fine-tuned checkpoint]         ← trained on full TOFU dataset
        │
        │  unlearning/main.py
        │  Config: unlearning/configs/phi-{GA,GD,NPO}.yaml
        ▼
[Unlearned checkpoint]          ← forget set removed via GA / GD / NPO
        │
        │  evaluation/lower_leakage_vis.py
        │  evaluation/bound_gap_analysis.py
        ▼
results/{experiment}/
├── responses/   – raw model text outputs (.npz)
├── scores/      – ROUGE-L score arrays (.npz)
└── plots/       – PDF figures (lower_bounds.pdf, gap_analysis.pdf)
```

---

## Module Details

### `finetuning/`

| File | Role |
|---|---|
| `main.py` | CLI entry point (Hydra) |
| `experiment.py` | Orchestrates the training workflow |
| `model.py` | Loads base model from HuggingFace |
| `preprocessing.py` | Formats TOFU dataset (question/answer tags, label masking) |
| `training.py` | HuggingFace Trainer wrapper |
| `configs/phi.yaml` | Hyperparameters (batch=16, lr=2e-5, 5 epochs) |
| `ds_config.json` | DeepSpeed ZeRO config for distributed training |

### `unlearning/`

| File | Role |
|---|---|
| `main.py` | CLI entry point (Hydra) |
| `experiment.py` | Orchestrates the unlearning workflow |
| `model.py` | Loads fine-tuned checkpoint |
| `dataloader.py` | Splits TOFU into forget / retain batches |
| `encoding.py` | Text encoding utilities |
| `unlearning.py` | Top-level unlearning orchestration |
| `unlearning_trainer.py` | Custom Trainer with three loss modes: **GA** (gradient ascent), **GD** (gradient difference), **NPO** (negative preference optimization) + optional entropy regularization |
| `prepare_deepspeed.py` | DeepSpeed initialisation |
| `configs/phi-{GA,GD,NPO}.yaml` | Per-method hyperparameters |

### `evaluation/`

| File | Role |
|---|---|
| `config.py` | Experiment registry and path helpers (`EXPERIMENTS` dict, `get_paths()`, `make_dirs()`) |
| `evaluate_leakage.py` | Core pipeline: sample responses → ROUGE-L scoring → conbo probability bounds |
| `models.py` | HuggingFace model/tokenizer loader (used by the HF sampling path) |
| `sampling.py` | `generate()` / `generate_batch()` (used by the HF sampling path) |
| `visualize.py` | `plot_lower_bounds()`, `plot_gap_analysis()` |
| `lower_leakage_vis.py` | Entry point: runs evaluation + saves lower-bound PDF |
| `bound_gap_analysis.py` | Entry point: runs evaluation + saves gap-analysis PDF |
| `sampling-demo.ipynb` | Interactive demo notebook |

#### Evaluation internals

```
lower_leakage_vis.py / bound_gap_analysis.py
        │
        ├── config.py          (load experiment hparams + resolve paths)
        ├── evaluate_leakage.py
        │       ├── vLLM path  → fast batched sampling (use_vllm: true)
        │       └── HF path    → models.py + sampling.py (fallback)
        │           └── rouge-score → ROUGE-L per response
        │               └── conbo   → expectation & std bounds (alpha=0.01)
        └── visualize.py       (plot + save PDF)
```

Results are cached in `results/{experiment}/scores/` so subsequent runs skip sampling.

---

## Adding a New Experiment

1. Register it in `evaluation/config.py` under `EXPERIMENTS`:
   ```python
   "my_experiment": {
       "model":       "org/my-unlearned-model",
       "tokenizer":   "org/base-tokenizer",
       "dataset":     "locuslab/TOFU",
       "dataset_split": "forget05",
       "use_vllm":    True,
       ...
   }
   ```
2. Run:
   ```bash
   cd evaluation
   python lower_leakage_vis.py --experiment my_experiment
   python bound_gap_analysis.py --experiment my_experiment --no-show
   ```

---

## Quick Reference

```bash
# 1. Fine-tune
cd finetuning && python3 main.py -m -cd=configs -cn=phi

# 2. Unlearn (choose one)
cd unlearning && python3 main.py -m -cd=configs -cn=phi-GA
cd unlearning && python3 main.py -m -cd=configs -cn=phi-GD
cd unlearning && python3 main.py -m -cd=configs -cn=phi-NPO

# 3. Evaluate
cd evaluation
python lower_leakage_vis.py   --experiment simnpo_forget05 --no-show
python bound_gap_analysis.py  --experiment simnpo_forget05 --no-show
```
