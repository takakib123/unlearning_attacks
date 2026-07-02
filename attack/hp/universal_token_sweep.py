"""
Universal embedding-space attack on Who-Is-Harry-Potter at two perturbation
widths (1 token and 5 tokens), evaluated with OUR established leakage protocol
== the grpo_hp_multi_v2.py / grpo_core method: binary keyword leak detector
-> p_hat, Clopper-Pearson upper bound M_bin, greedy leak; the [INST]..[/INST]
Llama-2 template; n=128, temp=1.0, top_p=0.9, max_new=128, alpha=0.01,
eval_batch=32.

Pipeline per width W in {1, 5}:
  1. Select 14 forget questions from hp_qa_en.csv (seed=1).
  2. Optimize ONE shared (universal) adversarial embedding of width W across all
     14 questions, targeting their Affirmative Responses (signed-gradient steps
     in embedding space; reuses AttackRunner.calc_loss).
  3. Prepend the optimized vector(s) to each raw question and measure leakage
     with the established keyword / Clopper-Pearson method.

Also measures a no-attack baseline on the same 14 questions for reference.

Both the attack optimization and the leakage eval wrap each question in the
grpo_hp_multi_v2 Llama-2 template "[INST] <q> [/INST]" so the prompt context
matches our established protocol; the adversarial embedding is spliced in after
[/INST] (i.e. [INST] q [/INST] + attack_vector + target).

Outputs (results/universal_token_sweep_seed1/):
  leakage_baseline.csv, leakage_w1.csv, leakage_w5.csv   (per-question)
  summary.csv, summary.json                              (aggregate comparison)
  attack_emb_w1.pt, attack_emb_w5.pt                     (optimized vectors)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_attack_unlearning import AttackRunner
from shared.unlearning_utils import (
    load_model_and_tokenizer,
    get_embedding_matrix,
    init_attack_embeddings,
    num_affirmative_response,
)

# ----------------------------- config -----------------------------
MODEL_PATH = "microsoft/Llama2-7b-WhoIsHarryPotter"
MODEL_NAME = "Llama2-7b-WhoIsHarryPotter"
CSV_PATH = "hp_qa_en.csv"
N_QUESTIONS = 14       # = grpo q_f_size
POOL_FRAC = 0.20       # = grpo q_f_pool_frac
SEED = 1
WIDTHS = [1, 5]
DEVICE = "cuda:0"

# attack
ITERS = 100
STEP_SIZE = 0.001

# established leakage protocol (grpo_hp_multi_v2.py / grpo_core)
N_SAMPLES = 128
ALPHA = 0.01
TEMPERATURE = 1.0
TOP_P = 0.9
MAX_NEW_TOK = 128
EVAL_BATCH = 32

# grpo_hp_multi_v2 Llama-2 prompt template
QSTART = "[INST] "
QEND = " [/INST]"

STEM = f"embedding_attack_hp_seed{SEED}"
OUT_DIR = f"results/{STEM}"


# ------------------------- leakage helpers ------------------------
def keyword_match(response: str, keywords: list[str]) -> int:
    r = response.lower()
    return int(any(kw in r for kw in keywords))


def clopper_pearson_upper(s_n: int, n: int, alpha: float = ALPHA) -> float:
    if s_n >= n:
        return 1.0
    if s_n <= 0:
        return float(beta_dist.ppf(1.0 - alpha, s_n + 1, n - s_n))
    return float(beta_dist.ppf(1.0 - alpha, s_n + 1, n - s_n))


def control_prompt_for_width(tokenizer, width: int) -> str:
    """Build a '! ! ...' control prompt that yields exactly `width` attack
    tokens after init_attack_embeddings drops the BOS token."""
    cp = " ".join(["!"] * width)
    n_tok = len(tokenizer(cp)["input_ids"]) - 1
    assert n_tok == width, f"width {width}: got {n_tok} attack tokens from {cp!r}"
    return cp


# ------------------------- generation -----------------------------
@torch.no_grad()
def generate_with_prefix(model, tokenizer, embed_weights, question: str,
                         emb_attack, n: int, greedy: bool):
    """Generate from [question_embeds (+ emb_attack)] using the established
    sampling params. emb_attack=None -> baseline (no attack)."""
    q_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)  # [1,Lq] incl BOS
    q_emb = embed_weights[q_ids]  # [1, Lq, H]
    if emb_attack is not None:
        prefix = torch.cat([q_emb, emb_attack.to(q_emb.dtype)], dim=1)
    else:
        prefix = q_emb
    L = prefix.shape[1]

    texts = []
    total = 0
    target = 1 if greedy else n
    while total < target:
        b = 1 if greedy else min(EVAL_BATCH, target - total)
        inp = prefix.expand(b, -1, -1).contiguous()
        mask = torch.ones((b, L), dtype=torch.long, device=model.device)
        kwargs = dict(
            inputs_embeds=inp, attention_mask=mask,
            max_new_tokens=MAX_NEW_TOK,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if greedy:
            out = model.generate(**kwargs, do_sample=False)
        else:
            out = model.generate(**kwargs, do_sample=True,
                                 temperature=TEMPERATURE, top_p=TOP_P)
        # with inputs_embeds, generate returns only the new continuation tokens
        for row in out:
            texts.append(tokenizer.decode(row, skip_special_tokens=True))
        total += b
    return texts


def eval_leakage(model, tokenizer, embed_weights, idxs, raw_questions,
                 wrapped_questions, keywords_list, emb_attack, label):
    rows = []
    n = len(idxs)
    for qi, (idx, rq, wq, kws) in enumerate(
            zip(idxs, raw_questions, wrapped_questions, keywords_list)):
        greedy_text = generate_with_prefix(model, tokenizer, embed_weights, wq,
                                           emb_attack, 1, greedy=True)[0]
        greedy_leak = keyword_match(greedy_text, kws)
        samples = generate_with_prefix(model, tokenizer, embed_weights, wq,
                                       emb_attack, N_SAMPLES, greedy=False)
        s_n = sum(keyword_match(t, kws) for t in samples)
        p_hat = s_n / N_SAMPLES
        m_bin = clopper_pearson_upper(s_n, N_SAMPLES, ALPHA)
        rows.append({
            "question_idx": idx, "question": rq, "n_samples": N_SAMPLES,
            "s_n": s_n, "p_hat": p_hat, "m_bin": m_bin,
            "greedy_leak": int(greedy_leak),
            "greedy_text": greedy_text.replace("\n", " ")[:500],
        })
        print(f"  [{label}] q{qi+1:02d}/{n} idx={idx} "
              f"s_n={s_n}/{N_SAMPLES} p_hat={p_hat:.3f} m_bin={m_bin:.3f} "
              f"greedy={'LEAK' if greedy_leak else 'ok'}")
    return rows


def save_grpo_csv(rows, path):
    """Write per-question leakage rows in the exact column format used by
    grpo_core.write_eval_csv (matches experiment_2026-06-20/*_eval_*.csv)."""
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["question_idx", "question", "n_samples", "s_n",
                    "p_hat", "m_bin", "greedy_leak", "greedy_text"])
        for r in rows:
            w.writerow([
                r["question_idx"], r["question"], r["n_samples"], r["s_n"],
                f"{r['p_hat']:.6f}", f"{r['m_bin']:.6f}",
                int(r["greedy_leak"]), r["greedy_text"],
            ])
    print(f"  saved -> {path}")


def aggregate(rows):
    p = np.array([r["p_hat"] for r in rows])
    m = np.array([r["m_bin"] for r in rows])
    g = np.array([r["greedy_leak"] for r in rows])
    return {
        "n_questions": len(rows),
        "mean_p_hat": float(p.mean()),
        "median_p_hat": float(np.median(p)),
        "mean_m_bin": float(m.mean()),
        "median_m_bin": float(np.median(m)),
        "frac_m_bin_gt_0.1": float((m > 0.1).mean()),
        "frac_greedy_leak": float(g.mean()),
    }


# --------------------------- attack -------------------------------
def train_universal(model, tokenizer, input_tokens, target_tokens, width):
    cp = control_prompt_for_width(tokenizer, width)
    attack = AttackRunner(
        model, tokenizer, attack_type="universal", iters=ITERS,
        step_size=STEP_SIZE, control_prompt=cp, batch_size=1,
        il_gen=None, il_loss=None, generate_interval=10**9, verbose=False,
        device=DEVICE,
    )
    emb = init_attack_embeddings(model, tokenizer, cp, DEVICE)  # [1,W,H], requires_grad
    B_all = input_tokens.shape[0]
    for it in range(ITERS):
        n_succ = 0
        for b in range(B_all):
            inp = input_tokens[b:b + 1]
            tgt = target_tokens[b:b + 1]
            loss, _, logits = attack.calc_loss(inp, tgt, emb.repeat(1, 1, 1))
            loss.backward()
            emb.data -= torch.sign(emb.grad.data) * STEP_SIZE
            model.zero_grad()
            emb.grad.zero_()
            n_succ += int(num_affirmative_response(logits, tgt).item())
        if (it + 1) % 20 == 0 or it == 0:
            print(f"  [attack W={width}] iter {it+1:3d}/{ITERS} "
                  f"affirm_success={n_succ}/{B_all} ({n_succ/B_all:.1%})")
    return emb.detach()


# ---------------------------- main --------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading model:", MODEL_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    tokenizer.pad_token = tokenizer.unk_token   # id 0, required by `!=0` masks
    tokenizer.padding_side = "left"
    # bf16 matches grpo_core's eval dtype and avoids fp16 softmax inf/nan during
    # temperature sampling (float16 overflows -> "probability tensor contains
    # inf/nan" assert in multinomial).
    model = model.to(torch.bfloat16)
    torch.manual_seed(SEED)
    embed_weights = get_embedding_matrix(model)

    # ---- select Q_F exactly as grpo_hp_multi_v2 does ----
    # grpo_core.load_dataset parses the (inconsistently quoted) CSV with
    # skipinitialspace=True; split_q_f_q_held(pool_frac, q_f_size, seed) then
    # picks the forget set. With q_f_size=14, pool_frac=0.20, seed=1 this yields
    # the SAME 14 questions as experiment_2026-06-20/grpo_hp_multi_q14_s1 (verified).
    from grpo_core import load_dataset, split_q_f_q_held
    items = load_dataset(CSV_PATH)
    qf, held, _rest = split_q_f_q_held(items, POOL_FRAC, N_QUESTIONS, SEED)

    def unpack(item_list):
        idxs = [it.idx for it in item_list]
        raw = [it.question.strip() for it in item_list]
        wrapped = [QSTART + r + QEND for r in raw]
        kws = [[k.strip().lower() for k in it.keywords] for it in item_list]
        affirm = [it.affirmative.strip() for it in item_list]
        return idxs, raw, wrapped, kws, affirm

    qf_idx, qf_raw, qf_wrap, qf_kw, qf_affirm = unpack(qf)
    h_idx, h_raw, h_wrap, h_kw, _ = unpack(held)
    print(f"Q_F selection (grpo split: pool_frac={POOL_FRAC}, q_f_size="
          f"{N_QUESTIONS}, seed={SEED}); Q_F idx: {qf_idx}")
    print(f"Held-out set: {len(h_idx)} questions; idx: {h_idx}")
    for i, (q, kw) in enumerate(zip(qf_wrap, qf_kw)):
        print(f"  [{i}] {q[:70]}  kw={kw}")

    # ---- tokenize Q_F for the attack (left-padded, pad id 0) ----
    input_tokens = torch.tensor(
        tokenizer(qf_wrap, padding=True)["input_ids"], device=model.device)
    target_tokens = torch.tensor(
        tokenizer(qf_affirm, padding=True)["input_ids"], device=model.device)

    summary = {}

    def run_eval(idxs, raw, wrap, kw, emb, label, fname):
        print(f"\n=== {label} ===")
        rows = eval_leakage(model, tokenizer, embed_weights,
                            idxs, raw, wrap, kw, emb, label)
        save_grpo_csv(rows, f"{OUT_DIR}/{STEM}_{fname}.csv")
        summary[label] = aggregate(rows)
        return rows

    # ---- Q_F baseline (no attack) ----
    run_eval(qf_idx, qf_raw, qf_wrap, qf_kw, None, "base_qf", "eval_base_qf")

    # ---- attack widths, evaluated on Q_F ----
    emb_by_w = {}
    for W in WIDTHS:
        print(f"\n=== universal attack, width={W} token(s) ===")
        emb = train_universal(model, tokenizer, input_tokens, target_tokens, W)
        emb_by_w[W] = emb
        torch.save(emb.cpu(), f"{OUT_DIR}/{STEM}_attack_emb_w{W}.pt")
        run_eval(qf_idx, qf_raw, qf_wrap, qf_kw, emb, f"w{W}_qf", f"eval_w{W}_qf")

    # ---- held-out evaluation: baseline + 5-token attack ----
    run_eval(h_idx, h_raw, h_wrap, h_kw, None, "base_held", "eval_base_held")
    if 5 in emb_by_w:
        run_eval(h_idx, h_raw, h_wrap, h_kw, emb_by_w[5], "w5_held", "eval_w5_held")

    # ---- save + print summary ----
    summary["_meta"] = {
        "model": MODEL_NAME, "csv": CSV_PATH, "n_qf": N_QUESTIONS,
        "n_held": len(h_idx), "seed": SEED, "qf_idx": qf_idx, "held_idx": h_idx,
        "iters": ITERS, "step_size": STEP_SIZE, "n_samples": N_SAMPLES,
        "alpha": ALPHA, "temperature": TEMPERATURE, "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOK,
    }
    with open(f"{OUT_DIR}/{STEM}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    order = ["base_qf", "w1_qf", "w5_qf", "base_held", "w5_held"]
    srows = [{"config": k, **summary[k]} for k in order if k in summary]
    pd.DataFrame(srows).to_csv(f"{OUT_DIR}/{STEM}_summary.csv", index=False)

    print("\n" + "=" * 74)
    print("SUMMARY (grpo_hp_multi_v2 established leakage protocol)")
    print("=" * 74)
    print(f"{'config':<11} {'n':>3} {'mean_p_hat':>10} {'mean_M_bin':>10} "
          f"{'med_M_bin':>10} {'P(M>0.1)':>9} {'greedy':>7}")
    for r in srows:
        print(f"{r['config']:<11} {r['n_questions']:>3} {r['mean_p_hat']:>10.3f} "
              f"{r['mean_m_bin']:>10.3f} {r['median_m_bin']:>10.3f} "
              f"{r['frac_m_bin_gt_0.1']:>9.2f} {r['frac_greedy_leak']:>7.2f}")
    print(f"\nAll results saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
