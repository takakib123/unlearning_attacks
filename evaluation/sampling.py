import torch


def generate(model, tokenizer, question, hparams, do_sample):
    """Single-response generation (greedy or one stochastic sample)."""
    prompt = [hparams["question_start_tag"] + question + hparams["question_end_tag"]]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    ).to(hparams["device"])

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        use_cache=True,
        max_length=hparams["max_length"],
        top_p=hparams["top_p"],
        temperature=hparams["temperature"],
    )

    outputs = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def generate_batch(model, tokenizer, question, hparams, num_samples):
    """
    Generate `num_samples` stochastic responses in a single batched forward pass.

    Repeats the prompt `num_samples` times to build a true batch, avoiding the
    attention-mask expansion bug present in newer transformers versions when
    using `num_return_sequences > 1`.

    Returns a list of `num_samples` decoded strings.
    """
    prompt = [hparams["question_start_tag"] + question + hparams["question_end_tag"]]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    ).to(hparams["device"])

    # Repeat the single prompt to form a batch — .repeat() ensures a contiguous
    # tensor that all CUDA kernels handle correctly.
    input_ids      = inputs.input_ids.repeat(num_samples, 1)       # (N, seq_len)
    attention_mask = inputs.attention_mask.repeat(num_samples, 1)  # (N, seq_len)
    prompt_len     = inputs.input_ids.shape[1]

    outputs = model.generate(
        input_ids,
        attention_mask = attention_mask,
        pad_token_id   = tokenizer.pad_token_id,
        eos_token_id   = tokenizer.eos_token_id,
        do_sample      = True,
        use_cache      = True,
        max_length     = hparams["max_length"],
        top_p          = hparams["top_p"],
        temperature    = hparams["temperature"],
    )

    outputs = outputs[:, prompt_len:]
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
