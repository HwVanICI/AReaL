import asyncio
import httpx
import re
import os
import httpx
import json

from areal.utils import logging

logger = logging.getLogger("WIKIReward")

# ===============================
# Helpers
# ===============================

def normalize_text(text):
    return (text or "").lower().strip()

_BOX = re.compile(r"\\boxed\s*\{")

def clean_final_answer(text):
    """Normalize model output to a clean final answer string."""
    text = (text or "").lower().strip()

    # Extract <answer> block
    html_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if html_matches:
        text = html_matches[-1].strip()
    
    # Fallback to "final answer: ..." style
    fa_matches = re.findall(r"final answer:\s*(.*)", text)
    if fa_matches:
        text = fa_matches[-1].strip()

    # Remove special tokens like <|im_end|>
    text = re.sub(r"<\|im_end\|>", "", text).strip()
    text = re.sub(r"<answer>", "", text).strip()
    text = re.sub(r"</answer>", "", text).strip()
    text = re.sub(r"final answer:", "", text).strip()
    

    return text


# ===============================
# HTTP client (shared)
# ===============================

_HTTP_CLIENT = None

def get_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=3000)
    return _HTTP_CLIENT


# ===============================
# Async runner
# ===============================

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ===============================
# Reward function (ENTRYPOINT)
# ===============================


def is_clean_answer(pred):
    # No XML / tags
    if "<" in pred or ">" in pred:
        return False
    
    # No long sentences (heuristic)
    if len(pred.split()) > 10:
        return False
    
    # No obvious placeholder patterns
    blacklist = ["final answer", "answer:", "solution", "unknown", "search()"]
    if any(b in pred.lower() for b in blacklist):
        return False

    return True

def wiki_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    pred = clean_final_answer(str(completions))
    gold = clean_final_answer(str(answer))
    
    # Fast path: exact match
    if pred == gold:
        return 1.0

    if not is_clean_answer(pred):
        return 0.0

    if len(pred) > len(gold) * 2:
        return 0.0

    if not pred:
        return 0.0

    url = os.getenv("WIKI_JUDGE_URL")

    with httpx.Client(timeout=30) as client:
        payload = {
            "model": "judge",
            "prompt": (
                "You are a strict evaluator.\n\n"

                "Your task is to determine whether the generated answer EXACTLY matches the gold answer in meaning.\n\n"

                "Rules:\n"
                "1. The generated answer must be a SHORT, DIRECT answer only.\n"
                "2. If the generated answer contains ANY extra words, explanations, sentences, or formatting, it is incorrect.\n"
                "3. If the generated answer contains angle brackets < >, XML/HTML tags, or placeholder phrases (e.g., 'final answer', 'solution', 'answer'), it is incorrect.\n"
                "4. If the generated answer is not the exact same entity as the gold answer, it is incorrect.\n"
                "5. Minor formatting differences (case, punctuation, articles like 'the') are allowed ONLY if the answer is otherwise identical.\n\n"

                "Respond ONLY with one of the following:\n"
                "<ANSWER>correct</ANSWER>\n"
                "<ANSWER>incorrect</ANSWER>\n\n"

                "Examples:\n\n"

                "Generated answer: yes\n"
                "Gold answer: yes\n"
                "Your answer: <ANSWER>correct</ANSWER>\n\n"

                "Generated answer: yes, that is correct\n"
                "Gold answer: yes\n"
                "Your answer: <ANSWER>incorrect</ANSWER>\n\n"

                "Generated answer: <final answer>yes</final answer>\n"
                "Gold answer: yes\n"
                "Your answer: <ANSWER>incorrect</ANSWER>\n\n"

                "Generated answer: James Smith\n"
                "Gold answer: Jane Doe\n"
                "Your answer: <ANSWER>incorrect</ANSWER>\n\n"

                f"Generated answer: {pred}\n"
                f"Gold answer: {gold}\n"
                "Your answer: <ANSWER>"
            ),
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 10,
            "min_tokens": 3,
        }

        r = client.post(url, json=payload)
        resp = r.json()["choices"][0]["text"].lower()
        # Extract text inside <ANSWER> tags
        resp = resp.split("</answer>")[0].split("<answer>")[-1].strip()

        return 1.0 if resp == "correct" else 0.0
