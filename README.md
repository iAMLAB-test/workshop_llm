# RAG Workshop – Dutch Pension-Law Demo

This repository accompanies an internal developer workshop on **Retrieval-Augmented Generation (RAG)** using Dutch pension-related law texts from _wetten.overheid.nl_ and an Azure-hosted **GPT-4o-mini** instance.

---

## 📂 Project layout

```
.
├── data/
│   ├── docs/                 # Raw law-text dumps (.txt) – one file per law
│   └── KG/
│       └── Law_graph.trig    # RDF/Linked-data knowledge-graph
├── RAG_workshop_dutch_pension_law.ipynb   # Main hands-on notebook
└── README.md
```

---

## 🚀 Quick start

From an empty source folder in which you will run the notebook, run the following commands:
1. **Clone & install**
   ```bash
   git init
   python -m venv .venv && source .venv/bin/activate
   git clone [repository](https://github.com/iAMLAB-test/workshop_llm)
   pip install -r requirements.txt
   ```

Using `config_template.yaml` fill it's contexts (provided seperately) and save under `config.yaml`:
2. **Configure environment**
   ```bash
   API_KEY: [Enter your API key here]
   AZURE_ENDPOINT: [Enter your Azure endpoint here]
   API_VERSION: [Enter your API version here]
   ```

---

## 📝 Notebook contents

| Section | What you’ll learn |
|---------|-------------------|
| **1. Layout inspection** | Peek into raw Dutch law texts and tokenise articles. |
| **2. Naïve retrieval**   | Compare whole-law prompting vs. article-loop prompting; record **token-cost, latency, accuracy**. |
| **3. Embeddings intro**  | Cosine-distance demo with different GPT models → `text-embedding-ada-002` → `mixedbread-ai/mxbai-embed-large`. |
| **4. Vector store**      | Build per-article FAISS index with MXBAI embeddings. |
| **5. RAG query chain**   | Retrieve top-5 articles → feed into GPT-4o-mini for retrieval questions. |
| **6. RAG using a knowledge graph**    | Advantages of using linked-data-based RAG. |
| **7. Wrap-up**           | Key takeaways & next steps. |

---

## 🛠 Dependencies

```
openai>=1.25
azure-identity
langchain-community
langchain-huggingface
tiktoken
faiss-cpu
transformers>=4.39
scikit-learn
torch
rdflib
networkx
```
