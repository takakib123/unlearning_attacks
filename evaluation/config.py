"""
config.py

Centralized experiment configuration.
Add a new entry to EXPERIMENTS to test a different model — everything else
(paths, evaluation, visualization) derives from the experiment name.
"""

import os

# ---------------------------------------------------------------------------
# Statistical significance level for conbo bounds
# ---------------------------------------------------------------------------
ALPHA = 0.01   # per-bound; total = 2*alpha (two-sided)

# ---------------------------------------------------------------------------
# Root directory for all saved outputs
# Anchored to the project root so this works regardless of invocation CWD.
# ---------------------------------------------------------------------------
RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ---------------------------------------------------------------------------
# Experiment registry
# Add new experiments here; use the key as --experiment on the CLI.
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "simnpo_forget05": {
        "model":                   "OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat",
        "checkpoint":              None,
        "tokenizer":               "meta-llama/Llama-2-7b-chat-hf",
        "question_start_tag":      "[INST] ",
        "question_end_tag":        " [/INST]",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget05",
        # vLLM settings
        "use_vllm":                True,
        "gpu_memory_utilization":  0.70,   # lower if other processes share the GPU
    },
        "simnpo_forget05_attacked": {
        "model":                   "OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat",
        "checkpoint":              None,
        "tokenizer":               "meta-llama/Llama-2-7b-chat-hf",
        "question_start_tag":      "[INST] ",
        "question_end_tag":        " [/INST]",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget05",
        # vLLM settings
        "use_vllm":                True,
        "gpu_memory_utilization":  0.70,   # lower if other processes share the GPU
    },
    "grad_ascent_forget01": {
        "model":                   "locuslab/llama2-7b_grad_ascent_1e-05_forget01",
        "checkpoint":              "checkpoint-3",
        "tokenizer":               "meta-llama/Llama-2-7b-chat-hf",
        "question_start_tag":      "[INST] ",
        "question_end_tag":        " [/INST]",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget01",
        # vLLM settings
        "use_vllm":                True,
        "gpu_memory_utilization":  0.60,
    },
    "kl_forget01": {
        "model":                   "locuslab/llama2-7b_KL_1e-05_forget01",
        "checkpoint":              "checkpoint-3",
        "tokenizer":               "meta-llama/Llama-2-7b-chat-hf",
        "question_start_tag":      "[INST] ",
        "question_end_tag":        " [/INST]",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget01",
        # vLLM settings
        "use_vllm":                True,
        "gpu_memory_utilization":  0.90,
    },
    "grad_diff_forget01": {
        "model":                   "locuslab/llama2-7b_grad_diff_1e-05_forget01",
        "checkpoint":              "checkpoint-3",
        "tokenizer":               "meta-llama/Llama-2-7b-chat-hf",
        "question_start_tag":      "[INST] ",
        "question_end_tag":        " [/INST]",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget01",
        # vLLM settings
        "use_vllm":                False,
        "gpu_memory_utilization":  0.90,
    },
    "idk_forget01": {
        "model":                   "locuslab/llama2-7b_idk_1e-05_forget01",
        "checkpoint":              "checkpoint-3",
        "tokenizer":               "meta-llama/Llama-2-7b-chat-hf",
        "question_start_tag":      "[INST] ",
        "question_end_tag":        " [/INST]",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget01",
        # vLLM settings
        "use_vllm":                False,
        "gpu_memory_utilization":  0.90,
    },
    "rmu_llama32_1b_forget10": {
        "model":                   "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer15_scoeff100_epoch5",
        "checkpoint":              None,
        "tokenizer":               "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer15_scoeff100_epoch5",
        "question_start_tag":      "<|start_header_id|>user<|end_header_id|>\n\n",
        "question_end_tag":        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget10",
        "use_vllm":                True,
        "gpu_memory_utilization":  0.80,
},
    "AltPO_llama32_1b_forget10": {
        "model":                   "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
        "checkpoint":              None,
        "tokenizer":               "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
        "question_start_tag":      "<|start_header_id|>user<|end_header_id|>\n\n",
        "question_end_tag":        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget10",
        "use_vllm":                True,
        "gpu_memory_utilization":  0.60,
},
    "NPO_llama32_1b_forget10": {
        "model":                   "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10",
        "checkpoint":              None,
        "tokenizer":               "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10",
        "question_start_tag":      "<|start_header_id|>user<|end_header_id|>\n\n",
        "question_end_tag":        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget10",
        "use_vllm":                True,
        "gpu_memory_utilization":  0.60,
},
    "SimNPO_llama32_1b_forget10": {
        "model":                   "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5",
        "checkpoint":              None,
        "tokenizer":               "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5",
        "question_start_tag":      "<|start_header_id|>user<|end_header_id|>\n\n",
        "question_end_tag":        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "num_samples":             128,
        "max_new_tokens":          128,
        "top_p":                   0.9,
        "temperature":             1.0,
        "device":                  "cuda",
        "dataset":                 "locuslab/TOFU",
        "dataset_split":           "forget10",
        "use_vllm":                True,
        "gpu_memory_utilization":  0.80,
},

}


# ---------------------------------------------------------------------------
# Path helpers — all paths derived from experiment name
# ---------------------------------------------------------------------------
def get_paths(experiment_name: str) -> dict:
    """Return a dict of all output paths for the given experiment."""
    base = os.path.join(RESULTS_BASE, experiment_name)
    return {
        "base":             base,
        "responses":        os.path.join(base, "responses", f"{experiment_name}_responses.csv"),
        "scores":           os.path.join(base, "scores",    f"{experiment_name}_scores.csv"),
        "plots":            os.path.join(base, "plots"),
        "lower_bounds_fig": os.path.join(base, "plots", "lower_bounds.pdf"),
        "gap_analysis_fig": os.path.join(base, "plots", "gap_analysis.pdf"),
    }


def make_dirs(experiment_name: str) -> dict:
    """Create output directories and return the paths dict."""
    paths = get_paths(experiment_name)
    for sub in ("responses", "scores", "plots"):
        os.makedirs(os.path.join(paths["base"], sub), exist_ok=True)
    return paths
