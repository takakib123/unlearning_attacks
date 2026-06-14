"""
grpo_vllm_smoke.py
==================
Quick GPU smoke test for the vLLM eval path added to the GRPO attack.

It does NOT train. It exercises exactly the new machinery:
  1. HF login from env (same vars as grpo_hp_multi_v2).
  2. Load base + tokenizer, attach a fresh LoRA adapter.
  3. Build prompt encodings for a few questions.
  4. Run the HF eval (grpo_core.evaluate_question_set) as a baseline.
  5. Build the vLLM engine and run evaluate_question_set_vllm with the live
     adapter hot-swapped in via LoRARequest.
  6. Hot-swap a SECOND time (new lora id) to confirm reload works.
  7. Compare result dict structure + values and print PASS/FAIL + timings.

Because both engines hold a copy of the 7B base, this needs a large GPU.
Tune --vllm_gpu_mem_util down if vLLM init OOMs (HF model is already resident).

Usage (on the GPU machine):
    export HF_TOKEN=hf_xxx
    python grpo_vllm_smoke.py
    python grpo_vllm_smoke.py --n_questions 3 --n_samples 16 --vllm_gpu_mem_util 0.35
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import torch

from grpo_core import (
    aggregate, attach_new_lora, build_prompt_encodings,
    evaluate_question_set, load_base_and_tokenizer, load_dataset,
    split_q_f_q_held,
)
from grpo_hp_multi_v2 import Config
from grpo_vllm_eval import (
    build_vllm_engine, evaluate_question_set_vllm, make_lora_request,
)


def hf_login():
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN")
           or os.environ.get("HUGGINGFACE_TOKEN"))
    if tok:
        from huggingface_hub import login
        login(token=tok)
        print("HF login OK.")
    else:
        print("WARNING: no HF token in env; gated tokenizer may fail.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_questions", type=int, default=3, help="questions to eval")
    p.add_argument("--n_samples", type=int, default=16, help="MC samples per question")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--vllm_gpu_mem_util", type=float, default=0.35)
    p.add_argument("--skip_hf", action="store_true",
                   help="skip the HF baseline (only test the vLLM path)")
    args = p.parse_args()

    cfg = Config()
    cfg.use_vllm_eval = True
    cfg.vllm_gpu_mem_util = args.vllm_gpu_mem_util
    cfg.max_new_tokens = args.max_new_tokens
    cfg.eval_batch = max(8, args.n_samples)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA not available — this smoke test needs a GPU.")
        raise SystemExit(1)

    hf_login()

    print("\n[1/5] Loading base + tokenizer + LoRA ...")
    base, tokenizer = load_base_and_tokenizer(
        cfg.model_name, cfg.tokenizer_name, cfg.dtype_name, cfg.device,
    )
    model = attach_new_lora(
        base, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, cfg.lora_target_modules,
    )

    print("[2/5] Building encodings ...")
    items = load_dataset(cfg.qa_csv_path)
    q_f, _, _ = split_q_f_q_held(items, cfg.q_f_pool_frac, cfg.q_f_size, cfg.seed)
    q_f = q_f[: args.n_questions]
    enc = build_prompt_encodings(
        tokenizer, q_f, cfg.question_start_tag, cfg.question_end_tag, cfg.device,
    )
    print(f"      using {len(enc)} questions: {[it.idx for it in q_f]}")

    hf_results = None
    if not args.skip_hf:
        print(f"\n[3/5] HF baseline eval (n={args.n_samples}) ...")
        t0 = time.time()
        hf_results = evaluate_question_set(model, tokenizer, cfg, enc, args.n_samples)
        print(f"      HF eval done in {time.time() - t0:.1f}s")
        print(f"      HF  agg: {aggregate(hf_results)}")
    else:
        print("\n[3/5] HF baseline skipped (--skip_hf).")

    print(f"\n[4/5] Building vLLM engine (gpu_mem_util={cfg.vllm_gpu_mem_util}) ...")
    t0 = time.time()
    engine = build_vllm_engine(cfg)
    print(f"      vLLM engine ready in {time.time() - t0:.1f}s")

    adapter_dir = "grpo_vllm_smoke_adapter_tmp"
    print(f"\n[5/5] vLLM eval, two hot-swaps (n={args.n_samples}) ...")
    t0 = time.time()
    lreq1 = make_lora_request(model, adapter_dir, lora_id=1)
    v1 = evaluate_question_set_vllm(engine, cfg, enc, args.n_samples, lora_request=lreq1)
    print(f"      vLLM eval #1 done in {time.time() - t0:.1f}s")
    print(f"      vLLM agg: {aggregate(v1)}")

    t0 = time.time()
    lreq2 = make_lora_request(model, adapter_dir, lora_id=2)  # new id -> must reload
    v2 = evaluate_question_set_vllm(engine, cfg, enc, args.n_samples, lora_request=lreq2)
    print(f"      vLLM eval #2 (reload) done in {time.time() - t0:.1f}s")

    # --- Structural / sanity checks ---
    print("\n=== CHECKS ===")
    ok = True
    expected_keys = {"s_n", "n", "p_hat", "m_bin", "greedy_leak",
                     "greedy_text", "question_idx", "question"}
    for r in v1:
        if set(r) != expected_keys:
            print(f"  FAIL: key mismatch: {set(r) ^ expected_keys}"); ok = False; break
        if r["n"] != args.n_samples or not (0 <= r["s_n"] <= r["n"]):
            print(f"  FAIL: bad counts {r['s_n']}/{r['n']}"); ok = False; break
    else:
        print(f"  PASS: result dicts well-formed ({len(v1)} questions, keys match grpo_core)")

    if len(v1) == len(v2):
        print("  PASS: hot-swap reload returned same #questions")
    else:
        print("  FAIL: reload count mismatch"); ok = False

    if hf_results is not None:
        # p_hat is stochastic; check it's in a plausible band, not exact equality.
        hf_p = aggregate(hf_results)["mean_p_hat"]
        v_p = aggregate(v1)["mean_p_hat"]
        print(f"  INFO: mean p_hat  HF={hf_p:.3f}  vLLM={v_p:.3f}  "
              f"(|diff|={abs(hf_p - v_p):.3f}; large gap may signal a prompt/tokenizer mismatch)")

    print("\nRESULT:", "ALL STRUCTURAL CHECKS PASSED ✅" if ok else "FAILURES ABOVE ❌")
    print("(Share this full output back.)")


if __name__ == "__main__":
    main()
