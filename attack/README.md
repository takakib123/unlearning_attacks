# Attack suite

Adversarial relearning / prompt attacks against unlearned models, evaluating whether
"forgotten" knowledge is actually gone or just suppressed. Organized by target dataset.

## Layout

| Dir | Contents |
|---|---|
| `shared/` | Dataset-agnostic infrastructure: GRPO training core, vLLM batched eval, extraction-strength metric, embedding-attack utilities, cross-experiment leak analysis. |
| `hp/` | Attacks on the Who-Is-Harry-Potter unlearned model (`hp_qa_en.csv`). |
| `tofu/` | Attacks on the TOFU forget05 unlearned model (SimNPO). |
| `wmdp/` | (planned) Attacks on WMDP-unlearned models, same structure as `hp/`/`tofu/`. |
| `logs/` | Run logs. |

Each dataset dir has its own `experiments/` subdirectory holding timestamped
`experiment_<date>/` run outputs (LoRA adapters, eval CSVs, rollouts).

## Running scripts

Output paths (`experiment_<date>/`, `results/`) and default CSV inputs are resolved
relative to the **current working directory**, not the script's location. Run scripts
from inside their dataset directory so outputs land in the right `experiments/` folder:

```
cd attack/hp && python grpo_hp_multi_v2.py ...
cd attack/tofu && python task5_grpo.py ...
```

Or pass explicit `--log_dir`/`--out_dir`/CSV path flags to redirect I/O elsewhere.

## Import convention

Scripts that use `shared/` modules bootstrap the attack root onto `sys.path`:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.grpo_core import ...
```

Same-directory imports (e.g. a `tofu/` script importing another `tofu/` module) stay
flat, since a script's own directory is on `sys.path` when run directly.

See `hp/README.md` and `tofu/README.md` for per-dataset details.
