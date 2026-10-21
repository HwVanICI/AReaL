def get_wiki_rl_dataset(path, split, tokenizer, max_length=None):
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=path)[split]


    one_shot = """
Use 'Search("put your search query here")' to search and end your answer with 'Final answer: <answer>'.
Follow this format:

1. Think step by step.
2. If needed, call Search("put your search query here"); keep the query concise—it should be suitable for Wikipedia.
3. If you call Search(), do NOT provide the final answer yet; you may search no more than 3 queries total. The top-k docs will be returned to you for each search.
4. Stop generating after your reasoning and search calls. The search results will be returned to you.
5. Once you have all the information and are confident, give the final answer only once: Final answer: <answer>.
6. Keep the final answer concise.
7. Search once. then give the final answer. If a search is part of the final answer, it will be incorrect automatically.
"""

    def process(x):
        # Append the user question cleanly without closing the one-shot example prematurely
        content = one_shot + f"\n<PROMPT>Question: {x['question']}</PROMPT>"
        messages = [{"role": "user", "content": content}]
        return {
            "messages": messages,
            "answer": x["answer"]
        }

    dataset = dataset.map(process)

    if max_length:
        dataset = dataset.filter(
            lambda x: len(tokenizer.encode(x["messages"][0]["content"])) <= max_length
        )

    return dataset