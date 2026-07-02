# Harry Potter attacks

Target: `microsoft/Llama2-7b-WhoIsHarryPotter` (unlearned model). Dataset:
`hp_qa_en.csv` / `hp_qa_en_fixed.csv` (forget-set QA pairs about Harry Potter).

## Attack families

- **GRPO relearning** (`grpo_hp_multi_v2.py`) — LoRA-based reinforcement relearning
  on a subset of forget questions (Q_F), evaluated for leak generalization to held-out
  questions (Q_held). Uses `shared/grpo_core.py` for the training loop and
  `shared/grpo_vllm_eval.py` for batched vLLM evaluation. `grpo_vllm_smoke.py` is a
  smaller smoke-test harness around the same `Config`.
- **Embedding-space attack** (`embedding_attack_unlearning.py`) — optimizes an
  adversarial soft-prompt embedding to elicit forgotten completions; built on
  `shared/unlearning_utils.py`. `universal_token_sweep.py` and `run_universal_1token.py`
  are drivers that sweep/pin specific perturbation widths (1 and 5 tokens) using a
  shared "universal" embedding across all forget questions.

## Eval protocol

Binary keyword leak detector -> `p_hat` -> Clopper-Pearson upper bound `M_bin`
(alpha=0.01), plus greedy-completion leak. Llama-2 `[INST]...[/INST]` template,
n=128 samples, temp=1.0, top_p=0.9, max_new=128.

## Outputs

`experiments/experiment_<date>/` — LoRA adapters, pre/post eval CSVs, training
progress, rollout dumps. Run scripts from `attack/hp/` (or pass `--log_dir`) so
output lands here rather than the caller's CWD.
