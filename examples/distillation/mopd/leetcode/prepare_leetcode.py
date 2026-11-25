import pandas as pd
from datasets import Dataset
import subprocess
import re
import ast

# Download dataset
train_url = "https://huggingface.co/datasets/newfacade/LeetCodeDataset/resolve/main/LeetCodeDataset-train.jsonl"
test_url = "https://huggingface.co/datasets/newfacade/LeetCodeDataset/resolve/main/LeetCodeDataset-test.jsonl"

# REQUIRED_MODULES = ["heapq", "collections", "math", "bisect", "itertools"]

ALLOWED_IMPORTS = {
    "math", "heapq", "itertools", "collections", "functools",
    "string", "bisect", "operator", "random", "datetime",
    "typing"
}
SUBSETS = ["Medium"]


FLOAT_ASSERT_RE = re.compile(
    r"assert\s+(.*?)==\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)"
)


SKIP_ASSERT_KEYWORDS = [
    "tree_node",
    "is_same_tree",
    "list_node",
    "is_same_list",
    "linked_list_to_list"
]


LIST_COMPARATOR = """
def _eq_list(a, b):
    if a == b:
        return True

    if isinstance(a, list) and isinstance(b, list):
        try:
            def normalize(x):
                if isinstance(x, list):
                    return tuple(sorted(normalize(i) for i in x))
                return x

            return sorted(normalize(i) for i in a) == sorted(normalize(i) for i in b)
        except:
            pass

    return False
"""


LIST_ASSERT_RE = re.compile(
    r"assert\s+(.*?)==\s*(\[[^\n]*\])"
)
def should_skip_assert(lhs: str, rhs: str) -> bool:
    text = lhs + " " + rhs
    return any(k in text for k in SKIP_ASSERT_KEYWORDS)

def fix_float_asserts(code: str) -> str:
    def repl(match):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()

        if should_skip_assert(lhs, rhs):
            return match.group(0)

        return f"assert abs({lhs} - ({rhs})) <= 1e-6"

    return FLOAT_ASSERT_RE.sub(repl, code)

def fix_list_asserts(code: str) -> str:
    def repl(match):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()

        # skip trees / linked structures
        if should_skip_assert(lhs, rhs):
            return match.group(0)

        return f"assert _eq_list({lhs}, {rhs})"

    return LIST_ASSERT_RE.sub(repl, code)

def ensure_modules(code: str) -> str:
    for mod in ALLOWED_IMPORTS:
        if f"{mod}." in code and f"import {mod}" not in code:
            code = f"import {mod}\n" + code
    return code

def is_inplace_problem(starter_code: str) -> bool:
    if not isinstance(starter_code, str):
        return False
    return "-> None" in starter_code

# remove unallowed imports from the testing code
# primarily just sortedcontainers
def filter_imports(code: str) -> str:
    tree = ast.parse(code)

    new_body = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
            if all(name in ALLOWED_IMPORTS for name in names):
                new_body.append(node)

        elif isinstance(node, ast.ImportFrom):
            module = node.module.split(".")[0] if node.module else ""
            if module in ALLOWED_IMPORTS:
                new_body.append(node)

        else:
            new_body.append(node)

    tree.body = new_body
    return ast.unparse(tree)


def clean_imports(code: str) -> str:
    try:
        code = filter_imports(code)
    except Exception:
        return ""
    # code = ensure_modules(code)
    return code

def download(url):
    subprocess.run(["wget", "-q", url, "-O", url.split("/")[-1]], check=True)


def add_test_check(row):
    test_str = row["tests"]
    entry = row["entry_point"]

    if not isinstance(test_str, str):
        return ""

    if "def test_check():" in test_str:
        return test_str

    if not test_str.endswith("\n"):
        test_str += "\n"

    # 1. floats
    test_str = fix_float_asserts(test_str)

    # 2. lists
    new_test_str = fix_list_asserts(test_str)

    needs_list_comparator = new_test_str != test_str

    test_str = new_test_str

    # 3. prepend comparator if needed
    if needs_list_comparator:
        test_str = LIST_COMPARATOR + "\n" + test_str

    # 4. append test_check
    test_str += f"""
def test_check():
    check({entry})
"""

    return test_str

def strip_use_this_format(text: str) -> str:
    marker = "use this format:"
    idx = text.lower().find(marker)
    if idx != -1:
        return text[:idx].rstrip()
    return text

def process(path, out_name):

    df = pd.read_json(path, lines=True)
    df = df.rename(columns={
        "problem_description": "question",
        "test": "tests"
    })

    print("\n=== Difficulty distribution ===")
    print(df["difficulty"].value_counts())

    df = df[df["difficulty"].isin(SUBSETS)]
    df = df[~df["starter_code"].apply(is_inplace_problem)]

    df["tests"] = df.apply(add_test_check, axis=1)

    df["imports"] = df["prompt"].apply(clean_imports)
    df["question"] = df["question"] + "\nThe following classes and imports are defined: \n" + \
        df["imports"].fillna("") + "\n\nUse this format for the solution: \n" + df["starter_code"].fillna("")
    # df["question"] = df["question"].apply(strip_use_this_format)
    df = df[["question", "tests", "imports"]]
    dataset = Dataset.from_pandas(df)
    dataset.to_parquet(out_name)
    print(f"Prepared {len(dataset)} rows in {out_name}")


# run
download(train_url)
download(test_url)

process("LeetCodeDataset-train.jsonl", "train.parquet")
process("LeetCodeDataset-test.jsonl", "test.parquet")