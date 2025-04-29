"""
Utils used in the workshop notebook.
"""

import glob
import os

import tiktoken
import yaml
from openai import AzureOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage

with open("config.yaml", "r") as f:
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


def gpt_4o_mini(
    user_message: str, system_message: str = "", tool_schema: dict = {}
) -> ChatCompletionMessage:
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
    return response.choices[0].message


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


def multiplier(x: float, y: float) -> float:
    return x * y
