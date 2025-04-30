"""
Utils used in the workshop notebook.
"""

import glob
import json
import os
import re
import textwrap

import tiktoken
import yaml
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion

# 1) Load the Azure OpenAI API key and endpoint from config.yaml, create the API object.
parent_dir = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(parent_dir, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
client = AzureOpenAI(
    api_key=config["API_KEY"],
    azure_endpoint=config["AZURE_ENDPOINT"],
    api_version=config["API_VERSION"],
)

# 2) Define your tool (function) schema
tool_schema = {
    "type": "function",
    "function": {
        "name": "multiplier",
        "parameters": {
            "type": "object",
            "description": "Multiply two numbers.",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "First number to multiply, must be integer or float.",
                },
                "y": {
                    "type": "integer",
                    "description": "First number to multiply, must be integer or float.",
                },
            },
            "required": ["x", "y"],
        },
    },
}

SYSTEM = """The text between triple backtics is a Dutch law text, I will ask retrieval-based questions about it: ```{law_text}```"""
PRICE_PER_1K_INPUT = 0.0003  # <-- adjust to your contract
PRICE_PER_1K_OUTPUT = 0.0012


def split_articles(law_text):
    """
    Splits a law text into a dict mapping 'Artikel X' → its body text.
    """
    pattern = re.compile(
        r"\n\s*(Artikel\s+\d+[A-Za-z]*\.?)"  # capture the header, e.g. "Artikel 1" or "Artikel 2a."
        r"(.*?)"  # lazily capture everything up to...
        r"(?=\n\s*Artikel\s+\d+[A-Za-z]*\.?|\Z)",  # the next header or end of string
        re.DOTALL,
    )
    result = {}
    for match in pattern.finditer(law_text):
        header = match.group(1).strip()
        body = match.group(2).strip()
        result[header] = body
    return result


def gpt_4o_mini(
    user_message: str, system_message: str = "", tool_schema: dict = {}
) -> ChatCompletion:
    """
    Function to call the Azure OpenAI API with the gpt-4o-mini model.
    Args:
        system_message (str): The system message to set the context.
        user_message (str): The user message to send to the model.
    Returns:
        str: The model's response.
    """
    if len(tool_schema) > 0:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # deployment name
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            tools=[tool_schema],
        )
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # deployment name
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
    return response


def count_tokens(text: str, encoding) -> int:
    """
    Return number of tokens in text using the provided tiktoken encoding.
    Args:
        text (str): The text to count tokens in.
        encoding: The tiktoken encoding to use.
    Returns:
        int: The number of tokens in the text.
    """
    return len(encoding.encode(text))


def count_tokens_in_docs():
    """
    Count the number of tokens in all text files in the data/docs directory.
    Returns:
        str: A string with the total number of tokens across all files.
    """
    # Choose your model here:
    # e.g. "gpt-3.5-turbo", "gpt-4", or use a fixed encoding like "gpt2"
    model = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model)

    total_tokens = 0
    for filepath in glob.glob("data/docs/*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        n_tokens = count_tokens(text, encoding)
        print(f"{os.path.basename(filepath)}: {n_tokens} tokens")
        total_tokens += n_tokens

    return f"\nTotal tokens across all files: {total_tokens}"


def num_tokens(text: str, enc=tiktoken.encoding_for_model("gpt-4o-mini")) -> int:
    """
    Returns the number of tokens in a string using the specified encoding.
    """
    return len(enc.encode(text))


def estimate_cost(in_tokens: int, out_tokens: int = 0) -> float:
    """
    Estimate the cost of a request to the OpenAI API based on the number of input and output tokens.
    """
    return (in_tokens / 1000) * PRICE_PER_1K_INPUT + (
        out_tokens / 1000
    ) * PRICE_PER_1K_OUTPUT


def show_metrics(
    label: str,
    in_tokens: int,
    out_tokens: int,
    elapsed: float,
):
    """
    Show performance metrics for an API call
    """
    metrics = {
        "scenario": label,
        "input tokens": in_tokens,
        "output tokens": out_tokens,
        "USD cost": round(estimate_cost(in_tokens, out_tokens), 6),
        "elapsed seconds": round(elapsed, 3),
    }
    print(json.dumps(metrics, indent=2))


def multiplier(x: int, y: int) -> int:
    """
    Multiplier function to be used as a tool in the API call.
    """
    return x * y


def wrap_at_spaces(text: str, width: int) -> str:
    """
    Wraps text at the nearest whitespace ≤ width.
    """
    return textwrap.fill(text, width=width)
