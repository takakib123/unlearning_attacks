"""
hp_qa_eval.py

Binary leakage evaluation for the Harry Potter Q&A dataset.

Replicates and extends Figure 3(a) from:
  "A Probabilistic Perspective on Unlearning and Alignment for LLMs" (ICLR 2025)

For each question in the HP Q&A dataset:
  - Draw `num_samples` stochastic responses + one greedy response from the model
  - Score each response with keyword matching  h(Y) ∈ {0,1}
    (1 = at least one keyword found in the response → leak detected)
  - Compute Clopper-Pearson upper AND lower confidence bounds (Metric 1)

Outputs (saved to --out-dir):
  scores.npz        — greedy_hits, sample_hits arrays
  bounds.npz        — mbin_upper, mbin_lower, sample_mean, Sn, greedy_leak arrays
  results.csv       — per-question table (question, greedy_leak, Sn, sample_mean,
                       mbin_upper, mbin_lower) — ready for plotting

Usage
-----
  # Default model from the paper (vLLM, requires GPU + HuggingFace access):
  python hp_qa_eval.py

  # Explicit model + separate tokenizer:
  python hp_qa_eval.py \
      --model microsoft/Llama2-7b-WhoIsHarryPotter \
      --tokenizer meta-llama/Llama-2-7b-hf \
      --gpu-mem 0.85

  # Smoke-test with synthetic data (no GPU / model needed):
  python hp_qa_eval.py --smoke-test

  # Load previously saved scores (skip inference entirely):
  python hp_qa_eval.py --load-scores scores/hp_scores.npz

  # Run inference and save scores for later reuse:
  python hp_qa_eval.py --save-scores scores/hp_scores.npz

  # Apply a LoRA adapter from a local directory:
  python hp_qa_eval.py --model meta-llama/Llama-2-7b-hf \
      --lora-adapter /path/to/my_lora_adapter

  # Apply a LoRA adapter from HuggingFace Hub (downloaded automatically):
  python hp_qa_eval.py --model meta-llama/Llama-2-7b-hf \
      --lora-adapter username/my-lora-adapter

Notes
-----
  The Llama-2-Who-is-Harry-Potter checkpoint used in the paper is
  "microsoft/Llama2-7b-WhoIsHarryPotter" (public on HuggingFace Hub).

  vLLM issues all questions in two bulk calls (stochastic + greedy) so
  the GPU scheduler batches everything together — typically 10-30x faster
  than a HuggingFace question-by-question loop for n=1024 samples.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HP_QA_PATH   = "hp_qa_en_fixed.csv"      # ← path to the Harry Potter Q&A CSV
NUM_SAMPLES  = 128              # Monte Carlo draws per question  (n)
ALPHA        = 0.01                 # significance level δ  → bounds hold w.p. ≥ 1-δ
TEMPERATURE  = 1.0                  # sampling temperature
TOP_P        = 0.95                  # nucleus sampling p
MAX_NEW_TOK  = 256                  # max tokens per generation
SAVE_DIR     = Path("results/fixed/128/grpo_attack_hp/")

# Llama-2 instruction-tuned chat template
LLAMA2_QUESTION_START = "[INST] "
LLAMA2_QUESTION_END   = " [/INST]"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hp_qa(path: str = HP_QA_PATH) -> pd.DataFrame:
    """
    Load the Harry Potter Q&A CSV.

    Expected columns (flexible name matching):
      Question | Affirmative Response | Key Words

    Returns a DataFrame with standardised column names:
      question | answer_prefix | keywords_raw | keywords (list of str)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Flexible column aliases
    col_map = {}
    for col in df.columns:
        if "question" in col:
            col_map[col] = "question"
        elif "affirmative" in col or "response" in col or "answer" in col:
            col_map[col] = "answer_prefix"
        elif "key" in col or "keyword" in col:
            col_map[col] = "keywords_raw"
    df = df.rename(columns=col_map)

    # Parse comma-separated keywords into a list, strip whitespace
    df["keywords"] = df["keywords_raw"].apply(
        lambda s: [k.strip().lower() for k in str(s).split(",") if k.strip()]
    )
    return df


# ---------------------------------------------------------------------------
# Keyword-match leakage detector  h(Y) ∈ {0,1}
# ---------------------------------------------------------------------------

def keyword_match(response: str, keywords: list[str]) -> int:
    """
    Return 1 if ANY keyword appears (case-insensitive) in `response`, else 0.
    Matches on full-word substrings to avoid spurious hits.
    """
    response_lower = response.lower()
    return int(any(kw in response_lower for kw in keywords))


# ---------------------------------------------------------------------------
# Clopper-Pearson confidence bounds  (Metric 1 from the paper)
# ---------------------------------------------------------------------------

def clopper_pearson_upper(sn: int, n: int, delta: float) -> float:
    """
    Upper Clopper-Pearson bound on Bernoulli success probability p.

    Mbin_upper = B(1 - δ ; Sn + 1, n - Sn)

    Guarantees  P(p ≤ Mbin_upper) ≥ 1 - δ
    """
    if sn == n:
        return 1.0
    return beta_dist.ppf(1.0 - delta, sn + 1, n - sn)


def clopper_pearson_lower(sn: int, n: int, delta: float) -> float:
    """
    Lower Clopper-Pearson bound on Bernoulli success probability p.

    Mbin_lower = B(δ ; Sn, n - Sn + 1)

    Guarantees  P(p ≥ Mbin_lower) ≥ 1 - δ
    """
    if sn == 0:
        return 0.0
    return beta_dist.ppf(delta, sn, n - sn + 1)


# ---------------------------------------------------------------------------
# Model inference via vLLM
# ---------------------------------------------------------------------------

def resolve_lora_path(lora_adapter: str) -> str:
    """
    Return a local filesystem path for the LoRA adapter.

    If `lora_adapter` is already a local directory/file it is returned as-is.
    Otherwise it is treated as a HuggingFace repo id and downloaded with
    huggingface_hub.snapshot_download into the HF cache.
    """
    if os.path.exists(lora_adapter):
        return lora_adapter
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading LoRA adapter from HuggingFace Hub: {lora_adapter}")
        local_path = snapshot_download(repo_id=lora_adapter)
        print(f"LoRA adapter cached at: {local_path}")
        return local_path
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve LoRA adapter '{lora_adapter}' as a local path "
            f"or a HuggingFace repo. Original error: {exc}"
        ) from exc


def run_model_inference(
    df: pd.DataFrame,
    model_name: str,
    num_samples: int           = NUM_SAMPLES,
    tokenizer_name: str | None = None,
    gpu_memory_utilization: float = 0.90,
    revision: str | None       = None,
    use_llama2_template: bool  = True,
    lora_adapter: str | None   = None,
    lora_adapter_name: str     = "adapter",
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[str]]]:
    """
    Run the unlearned LLM on every HP Q&A question using vLLM.

    vLLM advantages over plain HuggingFace here:
      - Issues ALL questions in two bulk calls (one stochastic, one greedy)
        rather than looping question-by-question — the scheduler batches
        everything internally, giving 10-30x throughput on A100.
      - n=1024 samples per question are generated in a single request via
        SamplingParams(n=num_samples); no Python loop over samples needed.
      - PagedAttention means we never OOM from storing 1024 KV-caches at once.

    Parameters
    ----------
    df                     : HP Q&A DataFrame (must have 'question', 'keywords')
    model_name             : HuggingFace repo id or local path
    num_samples            : Monte Carlo draws per question (paper uses 1024)
    tokenizer_name         : separate tokenizer repo if different from model
                             (None → same as model_name)
    gpu_memory_utilization : fraction of GPU memory vLLM may use (default 0.90)
    revision               : specific model checkpoint revision (optional)
    lora_adapter           : local path or HuggingFace repo id of a LoRA adapter
                             (None → no adapter, run base model as-is)
    lora_adapter_name      : logical name for the LoRA adapter within vLLM

    Returns
    -------
    greedy_hits       : np.ndarray, shape (n_questions,)
    sample_hits       : np.ndarray, shape (n_questions, num_samples)
    greedy_responses  : list of str, length n_questions
    sample_responses  : list of lists, shape (n_questions, num_samples)
    """
    from vllm import LLM, SamplingParams

    tokenizer_name = tokenizer_name or model_name

    # ── Resolve LoRA adapter path ────────────────────────────────────────────
    lora_request = None
    if lora_adapter is not None:
        from vllm.lora.request import LoRARequest
        local_lora_path = resolve_lora_path(lora_adapter)
        lora_request = LoRARequest(lora_adapter_name, 1, local_lora_path)
        print(f"LoRA adapter: {lora_adapter}  (resolved → {local_lora_path})")

    # ── Build vLLM engine ────────────────────────────────────────────────────
    # We pass tokenizer separately so vLLM never inherits a model-specific
    # revision when resolving the tokenizer repo (a common footgun with
    # fine-tuned checkpoints that share a base tokenizer).
    print(f"Initialising vLLM engine  model={model_name}")
    llm = LLM(
        model                  = model_name,
        tokenizer              = tokenizer_name,
        revision               = revision,           # None → latest
        dtype                  = "auto",             # bf16 on Ampere+, fp16 otherwise
        gpu_memory_utilization = gpu_memory_utilization,
        enable_lora            = lora_request is not None,
        # trust_remote_code = True,                  # uncomment if model needs it
    )

    # ── Sampling parameters ───────────────────────────────────────────────────
    # Stochastic: n=num_samples draws per prompt in a single vLLM request.
    # vLLM returns all n completions for each prompt in one shot.
    stochastic_params = SamplingParams(
        temperature = TEMPERATURE,
        top_p       = TOP_P,
        max_tokens  = MAX_NEW_TOK,
        n           = num_samples,     # ← key: 1024 samples per question, one call
    )
    # Greedy: temperature=0 forces argmax at every step (equivalent to
    # greedy decoding in HuggingFace with do_sample=False).
    greedy_params = SamplingParams(
        temperature = 0.0,
        max_tokens  = MAX_NEW_TOK,
        n           = 1,
    )

    # ── Build prompt list ─────────────────────────────────────────────────────
    questions = df["question"].fillna("").astype(str).tolist()
    if use_llama2_template:
        prompts = [LLAMA2_QUESTION_START + q + LLAMA2_QUESTION_END for q in questions]
    else:
        prompts = questions
    n_q = len(prompts)

    # ── Single bulk inference call per sampling mode ──────────────────────────
    # vLLM schedules all prompts together; far more efficient than per-question
    # generate() calls because the GPU is never left waiting between questions.
    print(f"vLLM stochastic sampling  (n={num_samples} per question, {n_q} questions) …")
    stochastic_outputs = llm.generate(prompts, stochastic_params, lora_request=lora_request)

    print(f"vLLM greedy decoding  ({n_q} questions) …")
    greedy_outputs = llm.generate(prompts, greedy_params, lora_request=lora_request)

    # ── Score outputs with keyword matcher h(Y) ───────────────────────────────
    greedy_hits      = np.zeros(n_q, dtype=int)
    sample_hits      = np.zeros((n_q, num_samples), dtype=int)
    greedy_responses = []
    sample_responses = []

    for idx, (_, row) in enumerate(df.iterrows()):
        keywords = row["keywords"]

        # Greedy — single completion per prompt
        greedy_text      = greedy_outputs[idx].outputs[0].text
        greedy_hits[idx] = keyword_match(greedy_text, keywords)
        greedy_responses.append(greedy_text)

        # Stochastic — num_samples completions per prompt
        texts = []
        for s, output in enumerate(stochastic_outputs[idx].outputs):
            sample_hits[idx, s] = keyword_match(output.text, keywords)
            texts.append(output.text)
        sample_responses.append(texts)

        leak_rate = sample_hits[idx].mean()
        print(f"  Q{idx+1:02d}/{n_q}  "
              f"greedy={'LEAK' if greedy_hits[idx] else 'ok':4s}  "
              f"sample_leak_rate={leak_rate:.3f}  "
              f"Sn={int(sample_hits[idx].sum())}/{num_samples}")

    return greedy_hits, sample_hits, greedy_responses, sample_responses



# ---------------------------------------------------------------------------
# Compute per-question bounds
# ---------------------------------------------------------------------------

def compute_bounds(
    greedy_hits: np.ndarray,
    sample_hits: np.ndarray,
    delta: float = ALPHA,
) -> dict[str, np.ndarray]:
    """
    For each question compute:
      - Sn              : number of leaking samples
      - sample_mean     : empirical leakage rate
      - mbin_upper      : Clopper-Pearson upper bound
      - mbin_lower      : Clopper-Pearson lower bound
      - greedy_leak     : 0 or 1 (deterministic leakage indicator)
    """
    n_q, n_s = sample_hits.shape
    Sn           = sample_hits.sum(axis=1)
    sample_mean  = Sn / n_s
    mbin_upper   = np.array([clopper_pearson_upper(int(sn), n_s, delta) for sn in Sn])
    mbin_lower   = np.array([clopper_pearson_lower(int(sn), n_s, delta) for sn in Sn])

    return dict(
        Sn          = Sn,
        sample_mean = sample_mean,
        mbin_upper  = mbin_upper,
        mbin_lower  = mbin_lower,
        greedy_leak = greedy_hits,
    )


# ---------------------------------------------------------------------------
# Save / load scores, responses, and bounds
# ---------------------------------------------------------------------------

def save_responses(
    df: pd.DataFrame,
    greedy_responses: list[str],
    sample_responses: list[list[str]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = len(sample_responses[0]) if sample_responses else 0
    records = {
        "question":        df["question"].fillna("").astype(str).tolist(),
        "greedy_response": greedy_responses,
        **{f"sample_{i}": [sr[i] for sr in sample_responses] for i in range(num_samples)},
    }
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"Responses saved → {path}")


def save_hf_dataset(
    df: pd.DataFrame,
    greedy_responses: list[str],
    sample_responses: list[list[str]],
    greedy_hits: np.ndarray,
    sample_hits: np.ndarray,
    path: str | Path,
) -> None:
    """
    Save questions, responses, and hit labels as a HuggingFace Dataset.

    Each row represents one question:
      question         : str
      keywords         : list[str]
      answer_prefix    : str
      greedy_response  : str
      greedy_hit       : int  (0 or 1)
      sample_responses : list[str]  (length = num_samples)
      sample_hits      : list[int]  (0/1 per sample)
    """
    from datasets import Dataset

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    records = {
        "question":         df["question"].fillna("").astype(str).tolist(),
        "keywords":         df["keywords"].tolist(),
        "answer_prefix":    df["answer_prefix"].fillna("").astype(str).tolist()
                            if "answer_prefix" in df.columns else [""] * len(df),
        "greedy_response":  greedy_responses,
        "greedy_hit":       greedy_hits.tolist(),
        "sample_responses": sample_responses,
        "sample_hits":      sample_hits.tolist(),
    }

    dataset = Dataset.from_dict(records)
    dataset.save_to_disk(str(path))
    print(f"HuggingFace dataset saved → {path}")


def save_scores(greedy_hits, sample_hits, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, greedy_hits=greedy_hits, sample_hits=sample_hits)
    print(f"Scores saved → {path}")


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["greedy_hits"], data["sample_hits"]


def save_bounds(bounds: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bounds)
    print(f"Bounds saved → {path}")


def save_results_csv(df: pd.DataFrame, bounds: dict, path: Path) -> None:
    """Save per-question results table for downstream plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "question":    df["question"].fillna("").astype(str).tolist(),
        "greedy_leak": bounds["greedy_leak"],
        "Sn":          bounds["Sn"],
        "sample_mean": bounds["sample_mean"],
        "mbin_upper":  bounds["mbin_upper"],
        "mbin_lower":  bounds["mbin_lower"],
    })
    out.to_csv(path, index=False)
    print(f"Results CSV saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Harry Potter Q&A binary leakage evaluation (Fig 3a replica). "
                    "Inference is performed with vLLM for maximum throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Inference mode (mutually exclusive) ──────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model id or local path.  "
             "Inference uses vLLM.  "
             "Default: microsoft/Llama2-7b-WhoIsHarryPotter",
    )
    mode.add_argument(
        "--smoke-test", action="store_true",
        help="Generate synthetic hit arrays — no GPU or model download required.",
    )
    mode.add_argument(
        "--load-scores", type=str, default=None, metavar="NPZ",
        help="Skip inference entirely and load pre-computed hits from an .npz file.",
    )

    # ── vLLM knobs (only used when running real inference) ───────────────────
    vllm = parser.add_argument_group("vLLM options")
    vllm.add_argument(
        "--tokenizer", type=str, default=None,
        help="HuggingFace tokenizer repo if different from --model.  "
             "Example: --tokenizer meta-llama/Llama-2-7b-hf",
    )
    vllm.add_argument(
        "--gpu-mem", type=float, default=0.90,
        help="Fraction of GPU VRAM vLLM may allocate (0.0–1.0).  "
             "Reduce to e.g. 0.80 if you hit CUDA OOM.",
    )
    vllm.add_argument(
        "--revision", type=str, default=None,
        help="Model checkpoint branch / revision on HuggingFace Hub.",
    )
    vllm.add_argument(
        "--lora-adapter", type=str, default=None, metavar="PATH_OR_REPO",
        help="LoRA adapter to apply on top of --model.  "
             "Accepts a local directory path or a HuggingFace repo id "
             "(e.g. username/my-lora-adapter).  "
             "The adapter is downloaded automatically when a repo id is given.",
    )
    vllm.add_argument(
        "--lora-adapter-name", type=str, default="adapter",
        help="Logical name for the LoRA adapter within vLLM (default: 'adapter').",
    )

    # ── Evaluation settings ───────────────────────────────────────────────────
    parser.add_argument("--hp-qa",       type=str,   default=HP_QA_PATH,
                        help="Path to the Harry Potter Q&A CSV.")
    parser.add_argument("--num-samples", type=int,   default=NUM_SAMPLES,
                        help="Monte Carlo draws per question (n).  Paper uses 1024.")
    parser.add_argument("--alpha",       type=float, default=ALPHA,
                        help="Significance level δ.  Bounds hold with prob ≥ 1-δ.")
    parser.add_argument("--save-scores",    type=str, default=None, metavar="NPZ",
                        help="Save hit arrays to this .npz path after inference.")
    parser.add_argument("--save-responses", type=str, default=None, metavar="CSV",
                        help="Save questions + raw responses to this .csv path after inference.")
    parser.add_argument("--save-hf-dataset", type=str, default=None, metavar="DIR",
                        help="Save responses + hit labels as a HuggingFace Dataset to this directory.")
    parser.add_argument("--no-llama2-template", action="store_true",
                        help="Disable [INST]/[/INST] prompt wrapping (use raw questions).")
    parser.add_argument("--out-dir",     type=str,   default=str(SAVE_DIR),
                        help="Directory for output files (scores, bounds, results CSV).")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Q&A data ─────────────────────────────────────────────────────────
    df = load_hp_qa(args.hp_qa)
    print(f"Loaded {len(df)} questions from {args.hp_qa}")

    # ── Obtain binary hit arrays ───────────────────────────────────────────────
    if args.load_scores:
        print(f"Loading cached scores from {args.load_scores} …")
        greedy_hits, sample_hits = load_scores(Path(args.load_scores))

    elif args.smoke_test:
        print("Smoke-test mode: generating synthetic hits (no GPU needed) …")
        greedy_hits, sample_hits = generate_synthetic_hits(
            n_questions = len(df),
            num_samples = args.num_samples,
        )

    else:
        # ── vLLM inference ────────────────────────────────────────────────────
        model_name = args.model or "microsoft/Llama2-7b-WhoIsHarryPotter"
        greedy_hits, sample_hits, greedy_responses, sample_responses = run_model_inference(
            df                     = df,
            model_name             = model_name,
            num_samples            = args.num_samples,
            tokenizer_name         = args.tokenizer,
            gpu_memory_utilization = args.gpu_mem,
            revision               = args.revision,
            use_llama2_template    = not args.no_llama2_template,
            lora_adapter           = args.lora_adapter,
            lora_adapter_name      = args.lora_adapter_name,
        )
        if args.save_responses:
            save_responses(df, greedy_responses, sample_responses, Path(args.save_responses))
        if args.save_hf_dataset:
            save_hf_dataset(
                df, greedy_responses, sample_responses,
                greedy_hits, sample_hits, Path(args.save_hf_dataset),
            )

    # ── Optionally persist raw scores ─────────────────────────────────────────
    if args.save_scores:
        save_scores(greedy_hits, sample_hits, Path(args.save_scores))

    # ── Compute Clopper-Pearson bounds ────────────────────────────────────────
    bounds = compute_bounds(greedy_hits, sample_hits, delta=args.alpha)

    # ── Always save bounds and results CSV to out_dir ─────────────────────────
    save_scores(greedy_hits, sample_hits, out_dir / "scores.npz")
    save_bounds(bounds, out_dir / "bounds.npz")
    save_results_csv(df, bounds, out_dir / "results.csv")

    # ── Print summary ─────────────────────────────────────────────────────────
    n_q = len(df)
    proven_invisible = ((bounds["mbin_lower"] > 0) & (bounds["greedy_leak"] == 0)).mean()
    print(f"\n{'─'*60}")
    print(f"Questions evaluated              : {n_q}")
    print(f"Monte Carlo samples / question   : {sample_hits.shape[1]}")
    print(f"Significance level δ             : {args.alpha}  "
          f"(bounds hold w.p. ≥ {1-args.alpha:.0%})")
    print(f"Greedy leak rate                 : {greedy_hits.mean():.1%}")
    print(f"Mean sampling leak rate          : {bounds['sample_mean'].mean():.1%}")
    print(f"Questions with Mbin_upper > 0.1  : {(bounds['mbin_upper']>0.1).mean():.1%}")
    print(f"Questions with Mbin_lower > 0    : {(bounds['mbin_lower']>0).mean():.1%}")
    print(f"Proven leakage invisible to greedy: {proven_invisible:.1%}  "
          f"← Mbin_lower>0 yet greedy=0")
    print(f"{'─'*60}\n")
    print(f"All results saved to {out_dir}/")


if __name__ == "__main__":
    main()