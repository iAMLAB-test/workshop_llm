"""
Utils used in the workshop notebook.
"""

import glob
import os
import re
import textwrap
from typing import Union

import tiktoken
import yaml
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse

# 1) Load the Azure OpenAI API key and endpoint from config.yaml, create the API object.
parent_dir = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(parent_dir, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client_decoder = AzureOpenAI(
    api_key=config["API_KEY_DECODER"],
    azure_endpoint=config["AZURE_ENDPOINT"],
    api_version=config["API_VERSION_DECODER"],
)
client_encoder = AzureOpenAI(
    api_key=config["API_KEY_ENCODER"],
    azure_endpoint=config["AZURE_ENDPOINT"],
    api_version=config["API_VERSION_ENCODER"],
)

SYSTEM_DECODER = """The text between triple backtics is a Dutch law text,
    I will ask retrieval-based questions about it: ```{law_text}```"""
SYSTEM_ENCODER = """The text between triple backtics is a Dutch law text, I will
    ask retrieval-based questions about it, embed it for that purpose: ```{law_text}```"""
PRICE_PER_1K_INPUT = 0.0003  # <-- adjust to your contract
PRICE_PER_1K_OUTPUT = 0.0012

# Encoder models and vector stores.
MXBAI_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={"device": "cpu"}
)
AZURE_EMBEDDINGS = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large",  # deployment name in your Azure portal
    openai_api_key=config["API_KEY_ENCODER"],
    azure_endpoint=config["AZURE_ENDPOINT"],
    api_version=config["API_VERSION_ENCODER"],
)
MXBAI_STORE_NONGRAPH = FAISS.load_local(
    os.path.join(
        parent_dir, f"src/data/vector_stores/mxbai-embed-large-v1_nongraph.index"
    ),
    MXBAI_EMBEDDINGS,
    allow_dangerous_deserialization=True,
)
AZURE_STORE_NONGRAPH = FAISS.load_local(
    os.path.join(
        parent_dir, f"src/data/vector_stores/text-embedding-3-large_nongraph.index"
    ),
    AZURE_EMBEDDINGS,
    allow_dangerous_deserialization=True,
)


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


def gpt_4o_mini(user_message: str, law_text: str = "") -> ChatCompletion:
    """
    Function to call the Azure OpenAI API with the gpt-4o-mini model.
    Args:
        law_text (str): The law text to use as context.
        user_message (str): The user message to send to the model.
    Returns:
        str: The model's response.
    """
    system_message = SYSTEM_DECODER.format(law_text=law_text)
    response = client_decoder.chat.completions.create(
        model="gpt-4o-mini",  # deployment name
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response


def text_embedding_3_large(text_list: list = []) -> CreateEmbeddingResponse:
    """
    Function to call the Azure OpenAI text_embedding_3_large embedder model.
    Args:
        text_list (list): The user message to send to the model.
    Returns:
        str: The model's response.
    """
    response = client_encoder.embeddings.create(
        model="text-embedding-3-large", input=text_list  # deployment name
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


def estimate_cost(in_tokens: int, out_tokens: int = 0) -> float:
    """
    Estimate the cost of a request to the OpenAI API based on the number of input and output tokens.
    """
    return (in_tokens / 1000) * PRICE_PER_1K_INPUT + (
        out_tokens / 1000
    ) * PRICE_PER_1K_OUTPUT


def llm_metrics(
    label: str,
    in_tokens: int,
    out_tokens: int,
    elapsed: float,
) -> dict:
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
    return metrics


def wrap_at_spaces(text: str, width: int) -> str:
    """
    Wraps text at the nearest whitespace ≤ width.
    """
    return textwrap.fill(text, width=width)


def topk(store, question: str, k: int = 5) -> list:
    """Return (docs, scores) for the k nearest neighbours."""
    return store.similarity_search_with_score(question, k=k)


def rag_executor(
    question: str, store=MXBAI_STORE_NONGRAPH
) -> Union[str, list, BaseCombineDocumentsChain]:
    """
    Uses RAG to answer a question about a law text.
    Args:
        question (str): The question to ask.
        store: The vector store to use for retrieval.
    Returns:
        str: The answer to the question.
        list: The most relevant documents retrieved.
    """
    # 1) search both stores
    hits = topk(store, question)

    # 2) keep the five best overall (lowest distance score)
    hits = sorted(hits, key=lambda x: x[1])[:5]
    retrieved_docs = [doc for doc, _ in hits]

    # 3) stuff those docs into a QA chain and ask the LLM
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",  # your chat deployment
        temperature=0,
        openai_api_key=config["API_KEY_DECODER"],
        azure_endpoint=config["AZURE_ENDPOINT"],
        api_version=config["API_VERSION_DECODER"],
    )
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    return (
        qa_chain.run(input_documents=retrieved_docs, question=question),
        list(retrieved_docs),
        qa_chain,
    )
