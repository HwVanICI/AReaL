# SPDX-License-Identifier: Apache-2.0

import os

from datasets import Value, load_dataset


def get_aime_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(path, "aime_train.parquet"),
            "test": os.path.join(path, "aime_test.parquet"),
        },
    )

    dataset = dataset[split]

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])

        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))

        return {
            "input_ids": seq_token,
            "loss_mask": loss_mask,
        }

    dataset = dataset.map(process, remove_columns=["question", "answer"])

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_aime_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(path, "aime_train.parquet"),
            "test": os.path.join(path, "aime_test.parquet"),
        },
    )

    dataset = dataset[split]

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process, remove_columns=["question"])

    dataset = dataset.cast_column("answer", Value("string"))
    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
