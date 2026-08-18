import os
import pickle

import streamlit as st
from dotenv import load_dotenv

# Vector Store & Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- RETRIEVERS (Safe multi-location imports) ---
from langchain_community.retrievers import BM25Retriever



# FlashRank Reranker
from langchain_community.document_compressors import FlashrankRerank
from groq import Groq, RateLimitError

load_dotenv()

# --- MODEL & PATH CONFIGURATIONS ---
#model_Name = "llama-3.3-70b-versatile"
model_Name = "openai/gpt-oss-120b"
model_Name_bkp = "llama-3.1-8b-instant"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "db")
CHUNKS_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "chunks")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="SAP BRIM Copilot", page_icon="⚡", layout="wide")
st.title("SAP BRIM AI Assistant")

with st.bottom:
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Created with ❤️ by Anurag Shukla"
        "</div>",
        unsafe_allow_html=True
    )

def reciprocal_rank_fusion(dense_docs, sparse_docs, k=60):
    """Combines BM25 and Vector DB results without needing EnsembleRetriever."""
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in reranked]

# --- HYBRID RETRIEVER LOADER ---
@st.cache_resource
def load_hybrid_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pickle_path = os.path.join(CHUNKS_PATH, "chunks.pkl")

    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH) or not os.path.exists(pickle_path):
        with st.spinner("Building vector database..."):
            import ingest
            ingest.build_vector_database()

    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    with open(pickle_path, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 15

    compressor = FlashrankRerank(
        model="ms-marco-TinyBERT-L-2-v2",
        top_n=7
    )

    return dense_retriever, bm25_retriever, compressor

# Load Hybrid components
try:
    dense_retriever, bm25_retriever, compressor = load_hybrid_retriever()
except Exception as e:
    st.error(f"Error loading database or retriever: {str(e)}")
    st.stop()

# --- GROQ CLIENT INITIALIZATION ---
@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

try:
    client = get_groq_client()
except Exception as e:
    st.error("Failed to connect to Groq API. Please check your GROQ_API_KEY in the .env file.")
    st.stop()

# --- CHAT HISTORY & USER INPUT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about SAP BRIM."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about SAP BRIM..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Executing Hybrid Search & Reranking...*")

        # --- RETRIEVAL PHASE ---
        # 1. Fetch dense and sparse candidates
        dense_docs = dense_retriever.invoke(user_query)
        sparse_docs = bm25_retriever.invoke(user_query)

        # 2. Merge using custom RRF (No EnsembleRetriever import required)
        raw_docs = reciprocal_rank_fusion(dense_docs, sparse_docs)

        # Rerank with FlashRank
        docs = compressor.compress_documents(documents=raw_docs, query=user_query)
        retrieved_context = "\n\n---\n\n".join([d.page_content for d in docs])

        response_placeholder.markdown("*Synthesizing complete response...*")

        # --- GENERATION PHASE ---
        system_instruction = (
            "You are a Principal SAP BRIM Technical Consultant (specializing in CM (Convergent Mediation), "
            "Convergent Charging, Convergent Invoicing, RAR, FSCM, and FI-CA).\n\n"
            "Your objective is to provide exhaustive, production-grade technical guides using ONLY "
            "the provided documentation context.\n\n"
            "CRITICAL GENERATION RULES:\n"
            "1. EXHAUSTIVE STEP-BY-STEP: When explaining configuration procedures, detail every step. "
            "Include exact file paths, transaction codes (T-codes), parameter names, tables, and CLI commands.\n"
            "2. NO SHORTCUTS: Do not summarize, omit, or skip configuration steps unless explicitly missing from the text.\n"
            "3. GAP TRANSPARENCY: If the retrieved manual text genuinely lacks crucial details to fulfill "
            "the user request, explicitly state: '⚠️ Documentation Gap: [Specify missing configuration or field]'.\n"
            "4. DOMAIN BOUNDARY: Only answer questions directly related to SAP BRIM. For out-of-scope "
            "topics, politely decline by stating: 'I am specialized in SAP BRIM. Please ask a BRIM-related question.'\n"
            "5. FORMATTING: Use bold markdown headers, numbered steps for execution sequences, "
            "and code blocks for commands or field mappings.\n"
            "6. EXTERNAL DATA CONDITION: If the query is related to SAP BRIM but context is limited, "
            "you may complement with standard SAP BRIM knowledge, but strictly remain within the SAP BRIM domain.\n"
            "7. STRICT DOMAIN FOCUS: Keep all responses strictly focused on SAP BRIM and S/4HANA Finance. Politely decline non-SAP queries.\n"
            "8. EXECUTIVE SUMMARY: Always conclude your response with a concise '### Executive Summary' section "
            "highlighting the core takeaways, key T-codes, or critical configuration steps."
        )

        prompt_payload = f"DOCUMENTATION CONTEXT:\n{retrieved_context}\n\nUSER QUESTION:\n{user_query}"

        def stream_groq_response(modelname):
            stream = client.chat.completions.create(
                model=modelname,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_payload}
                ],
                temperature=0.1,
                top_p=0.9,
                max_tokens=2048,
                stream=True
            )
            full_answer = ""
            for chunk in stream:
                delta_text = chunk.choices[0].delta.content
                if delta_text:
                    full_answer += delta_text
                    response_placeholder.markdown(full_answer + "▌")
            return full_answer

        # Execute generation with 70B -> 8B fallback
        try:
            full_answer = stream_groq_response(model_Name)
        except RateLimitError:
            st.toast("70B limit reached. Automatically switching to Llama-3.1 8B!")
            try:
                full_answer = stream_groq_response(model_Name_bkp)
            except Exception as fallback_err:
                response_placeholder.markdown(f"API Error: {str(fallback_err)}")
                full_answer = None
        except Exception as err:
            response_placeholder.markdown(f"API Error: {str(err)}")
            full_answer = None

        # Render Citations Footer
        if full_answer:
            sources_list = []
            for d in docs:
                source_name = os.path.basename(d.metadata.get("source", "SAP BRIM Manual"))
                page_num = d.metadata.get("page", None)

                if page_num:
                    entry = f"📄 **{source_name}** (Page {page_num})"
                else:
                    entry = f"📄 **{source_name}**"

                if entry not in sources_list:
                    sources_list.append(entry)

            citation_footer = "\n\n---\n### 📖 Verified SAP BRIM Documentation References:\n" + "\n".join(
                [f"* {s}" for s in sources_list]
            )

            final_response = full_answer + citation_footer
            response_placeholder.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
