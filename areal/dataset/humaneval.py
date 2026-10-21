from datasets import load_dataset
import os

def get_humaneval_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(path, "train.parquet"),
            "test": os.path.join(path, "test.parquet"),
        },
    )

    dataset = dataset[split]

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + ",".join(sample["tests"]) + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "tests"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_humaneval_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(path, "train.parquet"),
            "test": os.path.join(path, "test.parquet"),
        },
    )

    dataset = dataset[split]

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\n\nReturn ONLY Python code (no explanation), and include all necessary imports."
                "Put your answer inside a code block: ```python ... ```",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
