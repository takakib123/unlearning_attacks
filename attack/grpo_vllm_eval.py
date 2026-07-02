"""
grpo_vllm_eval.py
=================
vLLM-backed evaluation for the GRPO attack, a drop-in replacement for
``grpo_core.evaluate_question_set``.

Why: the HF ``model.generate`` eval loop runs one question at a time and
dominates wall-clock for n=128 pre/post evals and the periodic monitor.
vLLM batches the greedy pass and all n samples per question, which is far
faster on the same GPU.

How it coexists with training: ``grpo_hp_multi_v2.py`` keeps the HF training
model (weights + optimizer) resident. We spin up a *second* vLLM engine on
the same GPU with ``enable_lora=True`` and a capped ``gpu_memory_utilization``.
The current LoRA adapter is hot-swapped in each eval by saving it to disk and
issuing a fresh ``LoRARequest`` (monotonically increasing int id forces vLLM
to reload the updated weights). This loads a second copy of the 7B base, so
it needs a large GPU (e.g. A100-80GB).

Result dicts match grpo_core exactly:
    {s_n, n, p_hat, m_bin, greedy_leak, greedy_text, question_idx, question}
so ``aggregate`` / ``write_eval_csv`` / ``print_aggregate`` all work unchanged.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from grpo_core import clopper_pearson_upper, keyword_reward


def build_vllm_engine(cfg):
    """Construct the persistent vLLM engine. Reuse across all evals.

    Reads cfg attributes:
        model_name, tokenizer_name, lora_rank,
        vllm_gpu_mem_util, vllm_max_model_len, vllm_gpu

    Pins the engine to its own GPU (``cfg.vllm_gpu``), separate from the
    resident HF training model on ``cfg.train_gpu``. vLLM v1 runs its worker
    in a spawned subprocess, so temporarily restricting CUDA_VISIBLE_DEVICES
    around construction confines the engine (and its KV cache + 2nd base copy)
    to that single physical GPU. The parent's training CUDA context was already
    initialized on ``cfg.train_gpu`` and is unaffected; we restore the env after.
    """
    from vllm import LLM

    vllm_gpu = getattr(cfg, "vllm_gpu", None)
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vllm_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_gpu)
    try:
        return LLM(
            model=cfg.model_name,
            tokenizer=cfg.tokenizer_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="auto",                       # bf16 on A100, fp16 elsewhere
            max_model_len=cfg.vllm_max_model_len,
            gpu_memory_utilization=cfg.vllm_gpu_mem_util,
            enable_lora=True,
            max_lora_rank=cfg.lora_rank,
            max_loras=1,
            # Opt-in (default off): skip CUDA-graph capture. Saves memory and avoids
            # a common crash source when co-located with a resident training model.
            enforce_eager=getattr(cfg, "vllm_enforce_eager", False),
        )
    finally:
        if vllm_gpu is not None:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd


def make_lora_request(model, adapter_dir: str, lora_id: int):
    """Save the current PEFT adapter and wrap it in a fresh LoRARequest.

    A unique, increasing ``lora_id`` forces vLLM to reload the on-disk weights
    rather than serve a cached copy from a previous eval.
    """
    from vllm.lora.request import LoRARequest

    model.save_pretrained(adapter_dir)
    return LoRARequest(f"grpo_adapter_{lora_id}", lora_id, adapter_dir)


def evaluate_question_set_vllm(
    llm, cfg, encodings: List[dict], n_samples: int,
    lora_request=None, label: str = "",
) -> List[dict]:
    """Greedy + n probabilistic samples per question via batched vLLM.

    Reads cfg attributes:
        max_new_tokens, sampling_temperature, sampling_top_p, alpha
    """
    from vllm import SamplingParams

    if not encodings:
        return []

    prompts = [enc["prompt"] for enc in encodings]
    greedy_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=cfg.max_new_tokens,
    )
    sample_params = SamplingParams(
        n=n_samples,
        temperature=cfg.sampling_temperature,
        top_p=cfg.sampling_top_p,
        max_tokens=cfg.max_new_tokens,
    )

    t0 = time.time()
    greedy_out = llm.generate(prompts, greedy_params, lora_request=lora_request)
    sample_out = llm.generate(prompts, sample_params, lora_request=lora_request)

    results = []
    for enc, g, s in zip(encodings, greedy_out, sample_out):
        kws = enc["item"].keywords
        greedy_text = g.outputs[0].text
        greedy_leak = keyword_reward(greedy_text, kws)

        s_n = int(sum(keyword_reward(o.text, kws) for o in s.outputs))
        results.append({
            "s_n": s_n,
            "n": n_samples,
            "p_hat": s_n / n_samples,
            "m_bin": clopper_pearson_upper(s_n, n_samples, alpha=cfg.alpha),
            "greedy_leak": greedy_leak,
            "greedy_text": greedy_text,
            "question_idx": enc["item"].idx,
            "question": enc["item"].question,
        })

    if label:
        print(f"  [{label}] vLLM eval of {len(encodings)} questions "
              f"(n={n_samples}) in {time.time() - t0:.0f}s")
    return results
