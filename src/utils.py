"""
Utils used in the workshop notebook.
"""

import glob
import os
import re
import textwrap
import time
from typing import Union

import tiktoken
import yaml
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse
from rdflib import Dataset, Graph, Namespace, URIRef
from rdflib.namespace import RDF

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

# Encoder models and vector stores, can be sped up by using a GPU.
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
MXBAI_STORE_GRAPH = FAISS.load_local(
    os.path.join(
        parent_dir, f"src/data/vector_stores/mxbai-embed-large-v1_graph.index"
    ),
    MXBAI_EMBEDDINGS,
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


def text_embedding_3_large(text_list: list) -> CreateEmbeddingResponse:
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
    Count the number of tokens in all text files in the data/texts directory.
    Returns:
        str: A string with the total number of tokens across all files.
    """
    # Choose your model here:
    # e.g. "gpt-3.5-turbo", "gpt-4", or use a fixed encoding like "gpt2"
    model = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model)

    total_tokens = 0
    for filepath in glob.glob("data/texts/*.txt"):
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
) -> Union[str, list, BaseCombineDocumentsChain, dict]:
    """
    Uses RAG to answer a question about a law text.
    Args:
        question (str): The question to ask.
        store: The vector store to use for retrieval.
    Returns:
        str: The answer to the question.
        list: The most relevant documents retrieved.
        BaseCombineDocumentsChain: The chain used to combine documents.
        dict: Performance metrics.
    """
    start = time.time()
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
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}"
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
    performance_cache = {}
    with get_openai_callback() as cb:
        answer = qa_chain.invoke({"context": retrieved_docs, "question": question})
        performance_cache["in_tokens"] = cb.prompt_tokens
        performance_cache["out_tokens"] = cb.completion_tokens
        performance_cache["elapsed"] = time.time() - start
    return (answer, list(retrieved_docs), qa_chain, performance_cache)


# Graph utils
LAW_NODE_FORMAT = """
This Dutch Law article has title {label_desc} and the following sections between single backtics.
I will ask retrieval-based questions about it, encode it for that purpose.
`{sections}`
"""


def load_dataset(file_name: str) -> Dataset:
    """
    Load the dataset from the trig file and return it.

    Args:
        file_name (str): The name of the trig file without the extension.

    Returns:
        Dataset: The dataset object.
    """
    current_dir = os.path.dirname(__file__)
    dataset = Dataset()
    try:
        dataset.parse(
            os.path.abspath(os.path.join(current_dir, f"data/KG/{file_name}.trig")),
            format="trig",
        )
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {file_name}.trig. Error: {e}")
    return dataset


# Load all namespaces from the template graph.
DS = load_dataset("Law_graph")
name_space_mapper = {pr: Namespace(ns) for pr, ns in DS.namespaces()}
GRF = name_space_mapper["graph"]
ID = name_space_mapper["id"]
DEF = name_space_mapper["def"]
RDFS1 = name_space_mapper["rdfs1"]
LAW_ARTICLE = DEF.LawArticle
LABEL = RDFS1.label
HAS_SECTION = DEF.hasSection
HAS_DESCRIPTION = DEF.hasDescription
HAS_TEXT = DEF.hasText
HAS_CASES = DEF.hasCases
CROSS_REF = DEF.referencesTo
GRAPH_IRI = GRF.law_graph


def article_nodes(graph: Graph) -> dict:
    """
    Get the law nodes from the dataset and return their metadata and text representations.

    Args:
        graph (Graph): The graph object.
    Returns:
        dictionary containing tuples with article node text representations and the dataset.
    """

    art_dict = {}
    for law_node in graph.subjects(RDF.type, LAW_ARTICLE, unique=True):
        label = graph.value(law_node, LABEL)
        description = graph.value(law_node, HAS_DESCRIPTION)

        # Create a Dataset and a named graph for this law node (for metadata)
        dataset = Dataset()
        g = dataset.graph(GRAPH_IRI)
        g.add((law_node, RDF.type, LAW_ARTICLE))
        if label:
            g.add((law_node, LABEL, label))
        if description:
            g.add((law_node, HAS_DESCRIPTION, description))

        ## Title with description of the article
        if not description:
            label_desc = f"'{label}'"
        else:
            label_desc = f"'{label}' with description '{description}'"

        # Retrieve sections and cases of the article.
        sections = []
        for section in graph.objects(law_node, HAS_SECTION, unique=True):
            section_label = graph.value(section, LABEL)
            section_text = graph.value(section, HAS_TEXT)
            section_cases = list(graph.objects(section, HAS_CASES, unique=True))

            g.add((law_node, HAS_SECTION, section))
            if section_label:
                g.add((section, LABEL, section_label))
            if section_text:
                g.add((section, HAS_TEXT, section_text))
            if section_cases:
                for case in section_cases:
                    g.add((section, HAS_CASES, case))
            if len(section_cases) > 0:
                sections.append(
                    {
                        "iri": section,
                        "label": section_label,
                        "text": section_text,
                        "cases": "\n  - " + "\n  - ".join(section_cases),
                    }
                )
            else:
                sections.append(
                    {
                        "iri": section,
                        "label": section_label,
                        "text": section_text,
                    }
                )

        # Finally make metadata (subgraph) and the text representation of the article.
        section_lines = [
            f"Lid {s['label'].split("lid ")[-1]}: {s['text']}"
            for s in sections
            if s["label"] and s["text"]
        ]
        i_s = -1
        for s in sections:
            if s["label"] and s["text"]:
                i_s += 1
                if "cases" in s.keys():
                    section_lines[i_s] += s["cases"]
        text_representation = LAW_NODE_FORMAT.format(
            label_desc=label_desc,
            sections="\n".join(section_lines),
        )
        art_dict[str(law_node)] = text_representation, dataset

    return art_dict


def add_crossref_to_law_nodes(graph: Graph) -> dict:
    """
    Adds cross reference article text and metadata to the raw law nodes dictionary.
    Args:
        graph (Graph): The graph object.
    Returns:
        Dictionary containing tuples with article node text representations and the dataset.
    """
    node_dict = article_nodes(graph)
    art_dict = {}
    for law_node in graph.subjects(RDF.type, LAW_ARTICLE, unique=True):
        text_rep, dataset = node_dict[str(law_node)]
        cross_ref_count = 0
        for section in graph.objects(law_node, HAS_SECTION, unique=True):
            section_cross_ref_nodes = graph.objects(section, CROSS_REF, unique=True)
            for cross_ref_node in section_cross_ref_nodes:
                cross_ref_count += 1
                # In case the cross_ref_node does not refer to another article, skip it.
                if len(str(cross_ref_node).split("-")) < 3:
                    continue
                text_rep_cf, dataset_cf = node_dict[str(cross_ref_node)]
                if cross_ref_count == 1:
                    text_rep += f"\n This article contains cross-references to the following articles, each given between triple backtics, first article: ```{text_rep_cf}```."
                else:
                    text_rep += (
                        f"\n next cross-referenced article: ```{text_rep_cf}```."
                    )
                dataset = dataset + dataset_cf
        metadata = {
            "iri": law_node,
            "trig": dataset.serialize(format="trig"),
        }
        art_dict[str(law_node)] = (text_rep, metadata)
    return art_dict


def create_vector_store(
    encoder: HuggingFaceEmbeddings | AzureOpenAIEmbeddings = MXBAI_EMBEDDINGS,
) -> None:
    """
    Create a vector store from the specified graph names.

    Args:
        encoder: Either MXBAI or Azure OpenAI embeddings.
    """
    assert encoder in [
        MXBAI_EMBEDDINGS,
        AZURE_EMBEDDINGS,
    ], "Encoder must be either MXBAI or Azure OpenAI embeddings."

    if encoder == MXBAI_EMBEDDINGS:
        encoder_name = "mxbai-embed-large-v1"
    elif encoder == AZURE_EMBEDDINGS:
        encoder_name = "text-embedding-3-large"

    # Load the graph
    graph = load_dataset("Law_graph").graph(URIRef(GRF + "law_graph"))

    # Get the nodes from the graph
    art_dict = add_crossref_to_law_nodes(graph)
    metadatas = [m[1] for m in art_dict.values()]
    text_representations = [m[0] for m in art_dict.values()]

    # Create a vector store from the text representations and metadata
    vector_store = FAISS.from_texts(text_representations, encoder, metadatas)

    # Save the vector store as a FAISS index
    vector_store.save_local(
        os.path.join(parent_dir, f"src/data/vector_stores/{encoder_name}_graph.index")
    )
