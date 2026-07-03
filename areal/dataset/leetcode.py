# SPDX-License-Identifier: Apache-2.0

import os

from datasets import load_dataset

_PROMPT = """\n\n
You are solving a LeetCode-style problem.

STRICT REQUIREMENTS:
- Implement the method exactly as specified inside the provided class.
- Do NOT change the method name, arguments, or class name.
- Do NOT write a solve() function.
- Do NOT use input(), print(), or sys.stdin.
- Always RETURN the result. Never print.

- Use the provided data structures (e.g., ListNode, TreeNode) correctly.
- Do NOT use any non-standard python libraries like sortedcontainers
- Do NOT redefine helper classes unless required.

- Do not include unnecessary helper functions.
- Write clean and efficient code.

OUTPUT FORMAT:
- Return ONLY valid Python code.
- Do NOT include explanations.
- Do NOT include test code.
- Do NOT include a main block.
- Do NOT call the function.
- Do NOT output anything outside the class.
- Your entire response MUST be a single Python code block.
- Do NOT include any text outside the code block.

Example:

```python
    ...
```
The evaluator will call:
Solution().<method_name>(...)

"""


def get_leetcode_rl_dataset(
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
                "content": sample["question"] + _PROMPT,
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
