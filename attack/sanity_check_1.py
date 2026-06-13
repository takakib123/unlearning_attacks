"""
Step 1 prototype: Sampling-with-gradients via Gumbel-ST on Phi-1.5
===================================================================
Verifies the core mechanism for the Gumbel-Softmax LoRA attack described
in attack_protocol_v0.md §10 step 1:

  - inputs_embeds autoregressive loop (no input_ids during generation)
  - Gumbel-Softmax with Straight-Through estimator at each step
  - End-to-end forward + backward on a single short sequence
  - Gradients reach the LoRA parameters

The actual attack (step 2+) is NOT run here. This script is a correctness
check that has to pass before building anything on top of it.

Three checks, in order of importance:
  1. The autoregressive loop runs forward without error.
  2. backward() succeeds and produces nonzero gradients on LoRA params.
  3. (Sanity) A few AdamW steps reduce the loss -- confirms the gradient
     signal is meaningful, not just numerical noise.

Usage:
    pip install torch transformers peft
    python step1_sampling_with_gradients.py
"""


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "microsoft/phi-1_5"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Attack-side hyperparameters (protocol §5 defaults; small for the prototype)
LORA_RANK    = 4
LORA_ALPHA   = 16
LORA_TARGETS = ["q_proj", "v_proj"]   # native HF Phi attention projections
TAU          = 1.0                    # Gumbel temperature
KL_BETA      = 0.01                   # KL coefficient (matches §5 default)

# Single short sequence -- small enough to iterate on quickly
PROMPT     = "Question: Who are Harry Potter's best friends?\nAnswer:"
TARGET     = " Ron and Hermione"
GEN_LENGTH = 8

# Optimization sanity check
N_OPT_STEPS = 8
LR          = 1e-3


# ---------------------------------------------------------------------------
# Gumbel-Softmax with Straight-Through estimator
# ---------------------------------------------------------------------------
def gumbel_st_sample(logits: torch.Tensor, tau: float):
    """
    Sample one token from Categorical(softmax(logits)) using Gumbel-Softmax
    with the straight-through estimator.

    Returns:
        y_st   : [..., V]  one-hot in the forward pass, gradient of y_soft
                           in the backward pass (straight-through identity).
        y_soft : [..., V]  continuous Gumbel-softmax probabilities (used
                           directly in the reward; carries gradient).
    """
    # Sample standard Gumbel noise:  -log(-log(U)),  U ~ Uniform(0,1)
    u = torch.rand_like(logits, dtype=torch.float32).clamp(1e-9, 1 - 1e-9)
    gumbel = -torch.log(-torch.log(u)).to(logits.dtype)

    # Continuous, differentiable relaxation
    y_soft = F.softmax((logits + gumbel) / tau, dim=-1)

    # Hard one-hot via argmax of the perturbed logits
    idx    = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)

    # Straight-through identity: forward = y_hard, backward = grad(y_soft)
    y_st = y_hard + (y_soft - y_soft.detach())
    return y_st, y_soft


# ---------------------------------------------------------------------------
# Autoregressive rollout with inputs_embeds + Gumbel-ST
# ---------------------------------------------------------------------------
def rollout_with_gumbel_st(
    model,
    embed_layer: torch.nn.Embedding,
    prompt_ids:  torch.Tensor,    # [1, L_prompt]
    gen_length:  int,
    tau:         float,
):
    """
    Generate `gen_length` tokens autoregressively. At each step:
      1. Forward pass on the growing inputs_embeds sequence.
      2. Sample next token via Gumbel-ST from the last-position logits.
      3. Map the sampled one-hot to an embedding via y_st @ E.
      4. Append the new embedding to the sequence.

    KV caching is disabled deliberately -- re-running the full sequence each
    step is O(T^2) but trivially correct and easy to verify. The full attack
    will need a cache-aware version.

    Returns:
        soft_probs : list[gen_length] of [1, V] tensors (carry gradient)
        token_ids  : list[gen_length] of [1] long tensors (the sampled ids)
    """
    embeds = embed_layer(prompt_ids)        # [1, L_prompt, H]
    E      = embed_layer.weight             # [V, H]

    soft_probs, token_ids = [], []

    for _ in range(gen_length):
        out         = model(inputs_embeds=embeds, use_cache=False)
        next_logits = out.logits[:, -1, :]              # [1, V]

        y_st, y_soft = gumbel_st_sample(next_logits, tau)
        soft_probs.append(y_soft)
        token_ids.append(y_st.argmax(dim=-1))

        # Map (straight-through) one-hot to an input embedding and append
        next_embed = (y_st @ E).unsqueeze(1)            # [1, 1, H]
        embeds     = torch.cat([embeds, next_embed], dim=1)

    return soft_probs, token_ids


# ---------------------------------------------------------------------------
# Reward: log-probability of the target tokens under the soft distribution
# ---------------------------------------------------------------------------
def target_log_prob(soft_probs, target_ids):
    """
    Average log-probability assigned to the target token at each position,
    over min(gen_length, target_length) positions. This is the protocol §5
    reward restricted to the y* span.
    """
    T = min(len(soft_probs), target_ids.shape[0])
    log_p = soft_probs[0].new_zeros(())
    for t in range(T):
        p_t   = soft_probs[t][0, target_ids[t]].clamp(min=1e-12)
        log_p = log_p + torch.log(p_t)
    return log_p / T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[setup] device={DEVICE}  dtype={DTYPE}")
    print(f"[setup] loading {MODEL_ID} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation="eager",      # safe with inputs_embeds rollouts
    ).to(DEVICE)
    base_model.config.use_cache = False

    # -------- Inject LoRA on attention projections --------
    lora_cfg = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = 0.0,
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    embed_layer = model.get_input_embeddings()   # shared; used during rollout

    # -------- Build prompt / target token ids --------
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    target_ids = tokenizer(
        TARGET, return_tensors="pt", add_special_tokens=False
    ).input_ids[0].to(DEVICE)

    print(f"[setup] prompt length = {prompt_ids.shape[1]} tokens")
    print(f"[setup] target length = {target_ids.shape[0]} tokens "
          f"-> ids = {target_ids.tolist()}")
    print(f"[setup] gen_length    = {GEN_LENGTH} tokens\n")

    # -------- Reference (frozen) log-probs on the prompt, for KL --------
    # Disabling adapters gives us the base policy pi_{theta'}.
    with model.disable_adapter(), torch.no_grad():
        ref_out  = model(input_ids=prompt_ids, use_cache=False)
        ref_logp = F.log_softmax(ref_out.logits.float(), dim=-1)

    # -------- Optimizer on LoRA params only --------
    trainable   = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[setup] {len(trainable)} trainable tensors, "
          f"{n_trainable:,} scalars (LoRA only)\n")
    opt = torch.optim.AdamW(trainable, lr=LR)

    # ==========================================================
    # Verification loop
    # ==========================================================
    print("=== running prototype optimization ===")
    losses = []
    for step in range(N_OPT_STEPS):
        opt.zero_grad()

        # 1. Autoregressive rollout with Gumbel-ST
        soft_probs, token_ids = rollout_with_gumbel_st(
            model, embed_layer, prompt_ids, GEN_LENGTH, TAU
        )

        # 2. Reward: log-prob of the target under the soft distribution
        reward = target_log_prob(soft_probs, target_ids)

        # 3. KL penalty (cheap version: last prompt position only). The full
        #    protocol §5 KL is per-token across generated positions; here we
        #    just need a nonzero KL term that depends on the policy logits,
        #    to exercise the dual gradient pathway.
        policy_out  = model(input_ids=prompt_ids, use_cache=False)
        policy_logp = F.log_softmax(policy_out.logits.float(), dim=-1)
        policy_p    = policy_logp.exp()
        kl_lastpos  = (policy_p * (policy_logp - ref_logp))[0, -1].sum()

        # 4. Total loss: maximize reward (-> minimize -reward) + beta * KL
        loss = -reward + KL_BETA * kl_lastpos
        loss.backward()

        # 5. Gradient sanity check on the first step
        if step == 0:
            nonzero, total = 0, 0
            sample_norms = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                total += 1
                if p.grad is not None and p.grad.norm().item() > 0:
                    nonzero += 1
                    sample_norms.append((n, p.grad.norm().item()))

            print(f"[grad check] {nonzero}/{total} LoRA tensors have "
                  f"nonzero gradient")
            assert nonzero > 0, (
                "FAIL: no gradient reached any LoRA parameter. "
                "Check the Gumbel-ST + inputs_embeds path."
            )
            for n, g in sample_norms[:3]:
                print(f"             {n:55s}  ||grad|| = {g:.3e}")
            print("[grad check] PASS\n")

        opt.step()

        decoded = tokenizer.decode(torch.cat(token_ids))
        losses.append(loss.item())
        print(f"  step {step:2d}  loss={loss.item():+.4f}  "
              f"reward={reward.item():+.4f}  kl={kl_lastpos.item():.4f}  "
              f"sample={decoded!r}")

    # 6. Sanity: did the loss decrease?
    delta = losses[0] - losses[-1]
    print(f"\n[opt sanity] loss change over {N_OPT_STEPS} steps: "
          f"{losses[0]:+.4f} -> {losses[-1]:+.4f}  (delta = {delta:+.4f})")
    if delta > 0:
        print("[opt sanity] PASS: loss decreased")
    else:
        print("[opt sanity] WARNING: loss did not decrease -- Gumbel noise "
              "or tau may be too high for this short run, or LR too small")

    print("\n=== step 1 prototype complete ===")


if __name__ == "__main__":
    main()
