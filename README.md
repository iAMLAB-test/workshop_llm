# RAG Workshop – Dutch Pension-Law Demo

This repository accompanies an internal developer workshop on **Retrieval-Augmented Generation (RAG)** using Dutch pension-related law texts from _wetten.overheid.nl_ and an Azure-hosted **GPT-4o-mini** instance.

---

## 🚀 Quick start

From an empty source folder in which you will run the notebook, run the following commands:
1. **Clone & install**
   ```bash
   git init
   git clone https://github.com/iAMLAB-test/workshop_llm
   ```

2. Open the `src/RAG_workshop.ipynb` notebook and follow further instructions.

---

## 📝 Notebook contents

| Section | What you’ll learn |
|---------|-------------------|
| **1. Layout inspection** | Peek into raw Dutch law texts and tokenise articles. |
| **2. Naïve retrieval**   | Compare whole-law prompting vs. article-loop prompting; record **token-cost, latency, accuracy**. |
| **3. Embeddings intro**  | Cosine-distance demo with different encoders → `text-embedding-3-large` → `mixedbread-ai/mxbai-embed-large`. |
| **5. Vectore stores and RAG query chain**   | Retrieve top-5 articles → feed into GPT-4o-mini for retrieval questions. |
| **6. RAG using a knowledge graph**    | Advantages of using linked-data-based RAG. |
| **7. Wrap-up**           | Key takeaways & next steps. |

---

## 🛠 Dependencies

```
openai
langchain
langchain-community
langchain-huggingface
langchain_openai
langchain_core
tiktoken
rdflib
python-dotenv
faiss-cpu
numpy
scikit-learn
ipykernel
ipywidgets
```
