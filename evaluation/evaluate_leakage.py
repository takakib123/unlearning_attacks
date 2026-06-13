"""
evaluate_leakage.py

Sampling and leakage evaluation for a single experiment.

For each question in the TOFU forget set:
  - Sample `num_samples` stochastic responses and one greedy response
  - Score all responses with ROUGE-L against the ground truth
  - Compute conbo expectation bounds on E[ROUGE-L]

Saves under results/{experiment_name}/:
  responses/  — raw model text outputs (questions, greedy, stochastic)
  scores/     — ROUGE-L score arrays

Set "use_vllm": True in config.py to use the vLLM path (recommended).
Set "use_vllm": False to fall back to HuggingFace (slower, no external dep).

Usage
-----
  python evaluate_leakage.py --experiment simnpo_forget05
  python evaluate_leakage.py --experiment grad_ascent_forget01 --force
"""

import argparse
import atexit
import os
from concurrent.futures import ProcessPoolExecutor

import conbo
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from rouge_score import rouge_scorer as _rouge_scorer_lib
from tqdm.auto import tqdm

from config import ALPHA, EXPERIMENTS, make_dirs


# ---------------------------------------------------------------------------
# Parallel ROUGE-L helpers
# ---------------------------------------------------------------------------
# Each worker process initialises its own RougeScorer once (no GIL contention,
# no repeated construction overhead).

def _worker_init():
    global _SCORER
    _SCORER = _rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


def _score_pair(args: tuple) -> float:
    prediction, reference = args
    return _SCORER.score(reference, prediction)["rougeL"].fmeasure


# Shared pool — created lazily so forking only happens when needed.
_POOL: ProcessPoolExecutor | None = None
_POOL_WORKERS: int = 0


def _shutdown_pool() -> None:
    """Explicitly shut down the pool before interpreter teardown to avoid the
    weakref callback racing against module cleanup (AttributeError: 'NoneType'
    object has no attribute 'util')."""
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None


atexit.register(_shutdown_pool)


def _get_pool(n_workers: int) -> ProcessPoolExecutor:
    global _POOL, _POOL_WORKERS
    if _POOL is None or _POOL_WORKERS != n_workers:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
        )
        _POOL_WORKERS = n_workers
    return _POOL


def _rouge_single(prediction: str, reference: str) -> float:
    """Score one prediction against a reference (main-process scorer)."""
    if not hasattr(_rouge_single, "_scorer"):
        _rouge_single._scorer = _rouge_scorer_lib.RougeScorer(
            ["rougeL"], use_stemmer=True
        )
    return _rouge_single._scorer.score(reference, prediction)["rougeL"].fmeasure


def _batch_rouge(predictions: list, reference: str, n_workers: int = 0) -> np.ndarray:
    """Score every prediction against *reference* in parallel.

    Parameters
    ----------
    predictions : list of str
    reference   : str
    n_workers   : int
        Number of worker processes.  0 (default) = use all CPU cores.
        Pass 1 to force single-process execution (useful in interactive /
        already-parallelised contexts).
    """
    pairs = [(p, reference) for p in predictions]
    if n_workers == 1 or len(pairs) == 1:
        return np.array([_score_pair(a) for a in pairs])
    pool = _get_pool(n_workers or os.cpu_count() or 4)
    return np.array(list(pool.map(_score_pair, pairs, chunksize=max(1, len(pairs) // (pool._max_workers * 4)))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_prompt(question: str, hparams: dict) -> str:
    return hparams["question_start_tag"] + question + hparams["question_end_tag"]


# ---------------------------------------------------------------------------
# vLLM sampling path
# ---------------------------------------------------------------------------
def _run_vllm(hparams, dataset, n_questions):
    from vllm import LLM, SamplingParams
    from huggingface_hub import snapshot_download

    num_samples = hparams["num_samples"]

    # Download model and tokenizer to local cache separately so vLLM never
    # inherits the model revision when resolving the tokenizer repo.
    print("Downloading model weights …")
    model_path = snapshot_download(
        hparams["model"],
        revision=hparams.get("checkpoint"),
    )
    print("Downloading tokenizer …")
    tokenizer_path = snapshot_download(hparams["tokenizer"])

    llm = LLM(
        model                  = model_path,
        tokenizer              = tokenizer_path,
        dtype                  = "auto",
        gpu_memory_utilization = hparams.get("gpu_memory_utilization", 0.90),
    )

    stochastic_params = SamplingParams(
        temperature = hparams["temperature"],
        top_p       = hparams["top_p"],
        max_tokens  = hparams["max_new_tokens"],
        n           = num_samples,
    )
    greedy_params = SamplingParams(
        temperature = 0.0,
        max_tokens  = hparams["max_new_tokens"],
        n           = 1,
    )

    prompts = [_format_prompt(str(dataset[i]["question"] or ""), hparams) for i in range(n_questions)]

    # Single vLLM call per mode — scheduler batches everything internally
    print("vLLM: stochastic sampling …")
    stochastic_outputs = llm.generate(prompts, stochastic_params)

    print("vLLM: greedy decoding …")
    greedy_outputs = llm.generate(prompts, greedy_params)

    all_scores           = np.zeros((n_questions, num_samples))
    greedy_scores        = np.zeros(n_questions)
    greedy_responses     = np.empty(n_questions, dtype=object)
    stochastic_responses = np.empty((n_questions, num_samples), dtype=object)

    for qid in tqdm(range(n_questions), desc="Scoring"):
        answer = dataset[qid]["answer"]

        greedy_text            = greedy_outputs[qid].outputs[0].text
        greedy_responses[qid]  = greedy_text
        greedy_scores[qid]     = _rouge_single(greedy_text, answer)

        responses                  = [o.text for o in stochastic_outputs[qid].outputs]
        stochastic_responses[qid]  = responses
        all_scores[qid]            = _batch_rouge(responses, answer)

    return all_scores, greedy_scores, greedy_responses, stochastic_responses


# ---------------------------------------------------------------------------
# HuggingFace sampling path (fallback)
# ---------------------------------------------------------------------------
def _run_hf(hparams, dataset, n_questions):
    import torch
    from models import create_model
    from sampling import generate, generate_batch

    num_samples = hparams["num_samples"]
    batch_size  = hparams.get("sample_batch_size", num_samples)

    tokenizer, model = create_model(hparams)

    all_scores           = np.zeros((n_questions, num_samples))
    greedy_scores        = np.zeros(n_questions)
    greedy_responses     = np.empty(n_questions, dtype=object)
    stochastic_responses = np.empty((n_questions, num_samples), dtype=object)

    for qid in tqdm(range(n_questions), desc="Questions"):
        question = dataset[qid]["question"]
        answer   = dataset[qid]["answer"]

        greedy_out             = generate(model, tokenizer, question, hparams, do_sample=False)
        greedy_responses[qid]  = greedy_out[0]
        greedy_scores[qid]     = _rouge_single(greedy_out[0], answer)

        torch.manual_seed(qid)
        responses = []
        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)
            responses += generate_batch(model, tokenizer, question, hparams, n)

        stochastic_responses[qid] = responses
        all_scores[qid]           = _batch_rouge(responses, answer)

    return all_scores, greedy_scores, greedy_responses, stochastic_responses


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------
def run_evaluation(experiment_name: str, force: bool = False) -> dict:
    """
    Run the full sampling + scoring + bounding pipeline for one experiment.

    Returns a dict with keys: all_scores, greedy_scores, exp_lower, exp_upper,
    std_lower, std_upper, sample_mean, sample_var, questions, answers,
    author_ids, hparams, alpha.
    """
    hparams = EXPERIMENTS[experiment_name]
    paths   = make_dirs(experiment_name)

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset      = load_dataset(hparams["dataset"], hparams["dataset_split"])["train"]
    n_questions  = len(dataset)
    n_authors    = 20
    author_ids   = np.arange(n_questions) // (n_questions // n_authors)

    questions_text = np.array([dataset[i]["question"] for i in range(n_questions)])
    answers_text   = np.array([dataset[i]["answer"]   for i in range(n_questions)])

    # ── Check cache ───────────────────────────────────────────────────────────
    if not force and os.path.exists(paths["scores"]):
        print(f"Loading cached scores from {paths['scores']} …")
        df            = pd.read_csv(paths["scores"])
        greedy_scores = df["greedy_score"].values
        all_scores    = df[[c for c in df.columns if c.startswith("score_")]].values
    else:
        load_dotenv()
        login(token=os.environ["HUGGINGFACE_LOGIN_TOKEN"])

        if hparams.get("use_vllm", False):
            all_scores, greedy_scores, greedy_responses, stochastic_responses = \
                _run_vllm(hparams, dataset, n_questions)
        else:
            all_scores, greedy_scores, greedy_responses, stochastic_responses = \
                _run_hf(hparams, dataset, n_questions)

        # ── Save raw responses ────────────────────────────────────────────────
        num_samples = all_scores.shape[1]
        responses_df = pd.DataFrame({
            "question":        questions_text,
            "answer":          answers_text,
            "greedy_response": greedy_responses,
            **{f"response_{i}": stochastic_responses[:, i] for i in range(num_samples)},
        })
        responses_df.to_csv(paths["responses"], index=False)
        print(f"Responses saved to {paths['responses']}")

        # ── Save scores ───────────────────────────────────────────────────────
        scores_df = pd.DataFrame(
            all_scores,
            columns=[f"score_{i}" for i in range(all_scores.shape[1])],
        )
        scores_df.insert(0, "greedy_score", greedy_scores)
        scores_df.to_csv(paths["scores"], index=False)
        print(f"Scores saved to {paths['scores']}")

    # ── Compute bounds ────────────────────────────────────────────────────────
    n_questions = len(all_scores)
    exp_lower   = np.zeros(n_questions)
    exp_upper   = np.zeros(n_questions)
    std_lower   = np.zeros(n_questions)
    std_upper   = np.zeros(n_questions)
    sample_mean = np.zeros(n_questions)
    sample_var  = np.zeros(n_questions)

    for qid in range(n_questions):
        scores = all_scores[qid]
        sm, el, eu = conbo.expectation_bounds(scores, alpha=2 * ALPHA)
        _,  sl, su = conbo.std_bounds(scores,         alpha=2 * ALPHA)
        exp_lower[qid]   = el
        exp_upper[qid]   = eu
        std_lower[qid]   = sl
        std_upper[qid]   = su
        sample_mean[qid] = sm
        sample_var[qid]  = scores.var()

    return dict(
        all_scores    = all_scores,
        greedy_scores = greedy_scores,
        exp_lower     = exp_lower,
        exp_upper     = exp_upper,
        std_lower     = std_lower,
        std_upper     = std_upper,
        sample_mean   = sample_mean,
        sample_var    = sample_var,
        questions     = questions_text,
        answers       = answers_text,
        author_ids    = author_ids,
        hparams       = hparams,
        alpha         = ALPHA,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate leakage bounds for an unlearned LLM.")
    parser.add_argument(
        "--experiment", required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment name (key in config.EXPERIMENTS)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run sampling even if cached scores exist",
    )
    args = parser.parse_args()

    results = run_evaluation(args.experiment, force=args.force)
    print(f"\nDone. Bounds computed for {len(results['exp_lower'])} questions.")
    print(f"  Fraction with guaranteed leakage > 0.1: "
          f"{(results['exp_lower'] > 0.1).mean():.1%}")
