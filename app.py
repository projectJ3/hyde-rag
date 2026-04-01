import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv, find_dotenv
import gradio as gr

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv(find_dotenv())

# ── Settings ───────────────────────────────────────────────────────────────────
# HuggingFace embedding — runs free on HF Spaces, no Ollama needed
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Groq LLM — free tier, fast inference
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

Settings.chunk_size = 128
Settings.chunk_overlap = 30

# ── Qdrant Cloud client ────────────────────────────────────────────────────────
client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    port=443,
    prefer_grpc=False,
    https=True,
)

# ── Vector store ───────────────────────────────────────────────────────────────
vector_store = QdrantVectorStore(
    client=client,
    collection_name="hyde_collection"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ── Load documents & build / reuse index ──────────────────────────────────────
def load_documents():
    return SimpleDirectoryReader(
        input_dir="data",
        required_exts=[".pdf"]
    ).load_data()

def get_index():
    if not client.collection_exists(collection_name="hyde_collection"):
        logging.info("Collection not found — building index from documents...")
        documents = load_documents()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        logging.info("Index built and stored in Qdrant Cloud.")
    else:
        logging.info("Collection found — loading existing index from Qdrant Cloud.")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )
    return index

# Build index once at startup
logging.info("Initialising HyDE RAG pipeline...")
index = get_index()

# ── Query engines ──────────────────────────────────────────────────────────────
base_query_engine = index.as_query_engine(similarity_top_k=3)

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(base_query_engine, hyde)

logging.info("HyDE RAG pipeline ready.")

# ── Core query function ────────────────────────────────────────────────────────
def query_hyde(user_question: str, use_hyde: bool, history: list):
    """
    Run the user question through the selected query engine.
    Returns updated chat history for the Gradio Chatbot component.
    """
    if not user_question.strip():
        return history, ""

    try:
        if use_hyde:
            logging.info(f"[HyDE] Query: {user_question}")
            response = hyde_query_engine.query(user_question)
            mode_label = "**[HyDE mode]**"
        else:
            logging.info(f"[Base] Query: {user_question}")
            response = base_query_engine.query(user_question)
            mode_label = "**[Standard RAG mode]**"

        answer = str(response)

        # Retrieve source nodes for transparency
        source_info = ""
        if hasattr(response, "source_nodes") and response.source_nodes:
            sources = []
            for i, node in enumerate(response.source_nodes[:3], 1):
                score = f"{node.score:.3f}" if node.score else "N/A"
                snippet = node.text[:120].replace("\n", " ") + "..."
                sources.append(f"**Source {i}** (score: {score})\n> {snippet}")
            source_info = "\n\n---\n**Retrieved sources:**\n" + "\n\n".join(sources)

        full_response = f"{mode_label}\n\n{answer}{source_info}"
        history.append((user_question, full_response))

    except Exception as e:
        logging.error(f"Query failed: {e}")
        history.append((user_question, f"Error: {str(e)}"))

    return history, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="HyDE RAG System",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:

    gr.Markdown("""
    # HyDE RAG System
    **Hypothetical Document Embeddings** — an advanced retrieval technique that generates
    a hypothetical answer first, then uses it to retrieve more semantically relevant documents.

    > Powered by LlamaIndex · Groq (LLaMA 3.3 70B) · Qdrant Cloud · HuggingFace Embeddings
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
                bubble_full_width=False,
            )
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    label="Your question",
                    lines=2,
                    scale=4,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("Clear chat", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            use_hyde_toggle = gr.Checkbox(
                label="Enable HyDE",
                value=True,
                info="When ON, generates a hypothetical answer before retrieving. Usually improves quality."
            )
            gr.Markdown("""
            ### How HyDE works
            1. Your question comes in
            2. The LLM generates a **hypothetical answer** (even if factually imperfect)
            3. That hypothetical answer is embedded — not your question
            4. Semantically similar real documents are retrieved
            5. The LLM answers using the retrieved real context

            **Why it helps:** Real document embeddings are closer to answer-shaped text
            than question-shaped text. HyDE bridges that gap.

            ### Stack
            - **LLM:** Groq · LLaMA 3.3 70B
            - **Embeddings:** BAAI/bge-small-en-v1.5
            - **Vector DB:** Qdrant Cloud
            - **Framework:** LlamaIndex
            """)

    gr.Markdown("""
    ---
    <center><sub>Built with LlamaIndex + HuggingFace Spaces | HyDE: <a href="https://arxiv.org/abs/2212.10496">Gao et al. 2022</a></sub></center>
    """)

    # ── Event handlers ─────────────────────────────────────────────────────────
    submit_btn.click(
        fn=query_hyde,
        inputs=[question_input, use_hyde_toggle, chatbot],
        outputs=[chatbot, question_input],
    )

    question_input.submit(
        fn=query_hyde,
        inputs=[question_input, use_hyde_toggle, chatbot],
        outputs=[chatbot, question_input],
    )

    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, question_input],
    )

if __name__ == "__main__":
    demo.launch()
