import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Configuration & Paths
DB_PATH = "mediation-ai-tool/db/"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 2. Page Setup
st.set_page_config(page_title="MediationZone Offline Search", layout="wide")
st.title("MediationZone Offline Technical Search")
st.caption("Locally searching your 3,000+ page manual without any external cloud APIs")


# 3. Connect to your existing local DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


try:
    db = load_vector_db()
except Exception as e:
    st.error(f"Failed to connect to the database. Ensure your db/ folder exists!")
    st.stop()

# 4. Handle Search UI
if user_query := st.chat_input("Enter keywords (e.g., Aggregation Agent, EC creation)..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        st.write("**Scanning local vector database graphs...**")

        # Pull the top 3 most relevant text sections matching the meaning
        results = db.similarity_search(user_query, k=3)

        if not results:
            st.warning("No matching references found in the documentation.")
        else:
            st.success(f"Found {len(results)} exact matches inside your documentation:")

            # Display each chunk cleanly inside an expandable UI card
            for i, doc in enumerate(results, 1):
                page_num = doc.metadata.get('page', 'Unknown')

                with st.expander(f"Reference Match #{i} — From Page {page_num}", expanded=True):
                    st.markdown(doc.page_content)
                    st.caption(f"Source: {os.path.basename(doc.metadata.get('source', 'Manual'))}")