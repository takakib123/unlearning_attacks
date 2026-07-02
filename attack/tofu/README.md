# TOFU forget05 attacks

Target: `OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat` (unlearned), compared
against the base `locuslab/tofu_ft_llama2-7b`. Data: TOFU forget05 split (10 authors
x 20 QA pairs, via `tofu_forget05.py`) plus `tofu_forget05_keywords_DRAFT.csv`
(per-question leak keywords).

## Pipeline

Run in order:

1. `make_keyword_draft.py` — draft per-question keyword lists from the forget05
   reference answers -> `tofu_forget05_keywords_DRAFT.csv`.
2. `task3_dump.py` — dump baseline completions and gate questions via the leak oracle.
3. `task4_preeval.py` — full probabilistic pre-eval (n=128) on Q_F and Q_held for a
   given model (base or unlearned).
4. `task5_grpo.py` — GRPO relearning attack (LoRA) trained on Q_F only, then post-eval
   on Q_F and Q_held for generalization.
5. `add_es.py` — augment existing eval CSVs with the exact Extraction Strength metric
   without re-running sampling.

## Leak oracle & metrics

`tofu_oracle.py` defines the binary leak signal: keyword match OR ROUGE-L recall
>= 0.5, with a degeneracy guard against empty/repetitive completions. `tofu_eval.py`
reports `p_hat`, Clopper-Pearson `M_bin`, ROUGE-L bound `M_mu`/`M_sigma`, greedy leak,
and extraction strength (`shared/extraction.py`).

## Outputs

`experiments/experiment_<date>/` — LoRA adapters, `task4_pre_*`/`task5_*` CSVs,
rollout dumps. Run scripts from `attack/tofu/` (or pass `--out_dir`/explicit CSV
paths) so output and input CSVs resolve here rather than the caller's CWD.
