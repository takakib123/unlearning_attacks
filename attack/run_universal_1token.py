"""
Run the embedding-space attack from embedding_attack_unlearning.py in the
"1 token Universal" configuration on the Who-Is-Harry-Potter unlearned model.

Universal  -> one shared adversarial embedding is optimized across ALL forget
              questions (vs. a per-sample perturbation in the "individual" mode).
1 token    -> control_prompt = "!" tokenizes (after dropping BOS) to a single
              attack token, so the universal perturbation is exactly 1 token wide.

This is a thin driver around the existing AttackRunner so the original
embedding_attack_unlearning.py is left untouched. It also fixes the tokenizer
to the Llama-2 convention (pad_token = unk_token = id 0, left padding) that the
rest of the code assumes via its `tokens != 0` padding masks -- the bundled
load_tokenizer() only applies that for paths containing "llama-2", which this
model path ("...Llama2...") does not match.
"""

from embedding_attack_unlearning import AttackRunner
from unlearning_utils import load_model_and_tokenizer, load_dataset_and_dataloader, save_results

MODEL_PATH = "microsoft/Llama2-7b-WhoIsHarryPotter"
MODEL_NAME = "Llama2-7b-WhoIsHarryPotter"
DATASET_NAME = "hp_qa_en"
SEED = 42
TEST_SPLIT = 0
SHUFFLE = False

attack_config = {
    "attack_type": "universal",          # <-- universal (shared perturbation)
    "iters": 100,
    "step_size": 0.001,
    "control_prompt": "!",               # <-- 1 attack token after BOS is dropped
    "batch_size": 1,
    "early_stopping": False,
    "il_gen": "all",
    "il_loss": None,
    "generate_interval": 10,
    "num_tokens_printed": 100,
    "verbose": True,
}


def main():
    print("loading model:", MODEL_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # Llama-2 padding convention required by the `tokens != 0` masks elsewhere.
    tokenizer.pad_token = tokenizer.unk_token   # id 0
    tokenizer.padding_side = "left"
    print(f"tokenizer fixed: pad_token={tokenizer.pad_token!r} (id "
          f"{tokenizer.pad_token_id}), padding_side={tokenizer.padding_side}")

    import torch
    torch.manual_seed(SEED)

    attack = AttackRunner(model, tokenizer, **attack_config)

    # sanity: confirm the universal perturbation is exactly 1 token wide
    n_attack_tok = len(tokenizer(attack_config["control_prompt"])["input_ids"]) - 1
    print(f"universal attack width = {n_attack_tok} token(s)")
    assert n_attack_tok == 1, f"expected 1 attack token, got {n_attack_tok}"

    print("loading dataset:", DATASET_NAME)
    _, _, dl_train, dl_test = load_dataset_and_dataloader(
        attack.tokenizer,
        dataset_name=DATASET_NAME,
        batch_size=attack.batch_size,
        test_split=TEST_SPLIT,
        shuffle=SHUFFLE,
        device=model.device,
    )

    print("starting universal 1-token attack")
    result_dict = attack.attack(DATASET_NAME, dl_train, dl_test)
    save_results(result_dict, attack_config, MODEL_NAME, DATASET_NAME,
                 SHUFFLE, SEED, TEST_SPLIT)
    return result_dict


if __name__ == "__main__":
    main()
