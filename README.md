# HyDE RAG System

**Hypothetical Document Embeddings** — an advanced retrieval-augmented generation technique that generates a hypothetical answer first, then uses it to retrieve more semantically relevant documents.

## Stack

| Component    | Technology                   |
|-------------|------------------------------|
| **LLM**     | Groq · LLaMA 3.3 70B        |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HuggingFace) |
| **Vector DB**  | Qdrant Cloud                |
| **Framework**  | LlamaIndex                  |
| **UI**         | Gradio                      |

## How HyDE Works

1. Your question comes in
2. The LLM generates a **hypothetical answer** (even if factually imperfect)
3. That hypothetical answer is embedded — not your question
4. Semantically similar **real** documents are retrieved
5. The LLM answers using the retrieved real context

**Why it helps:** Document embeddings are closer to answer-shaped text than question-shaped text. HyDE bridges that gap.

## Local Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key
```

Run:

```bash
python app.py
```

## Deployment

Deployed on [HuggingFace Spaces](https://huggingface.co/spaces) with Gradio SDK (CPU basic, free tier).

---

> Built with LlamaIndex + HuggingFace Spaces | HyDE: [Gao et al. 2022](https://arxiv.org/abs/2212.10496)
