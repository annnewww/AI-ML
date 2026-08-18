import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq, RateLimitError

load_dotenv()
model_Name = "llama-3.3-70b-versatile"
model_Name_bkp = "llama-3.1-8b-instant"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "db")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(page_title="SAP BRIM Copilot", page_icon="", layout="wide")
st.title("SAP BRIM AI Assistant")
with st.bottom:
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Created with ❤️ by Anurag Shukla"
        "</div>",
        unsafe_allow_html=True
    )
#st.caption("Context-grounded assistant powered by your MediationZone Documentation & Gemini 2.0")
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if DB directory exists or is empty
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        with st.spinner(" Building vector database for first-time setup... Please wait 1-2 minutes."):
            import ingest
            ingest.build_vector_database()
            st.success(" Database created successfully!")

    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


# Load database (will build on 1st load, then load instantly for everyone else)
try:
    db = load_vector_db()
except Exception as e:
    st.error(f"Error loading database: {str(e)}")
    st.stop()

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

try:
    client = get_groq_client()
except Exception as e:
    st.error("Failed to connect to Groq API. Please check your GROQ_API_KEY in the .env file.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about SAP BRIM."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about SAP BRIM..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Searching local manual vector graph...*")

        # --- RETRIEVAL PHASE ---
        # Get top 7 closest text chunks from local vector db
        docs = db.max_marginal_relevance_search(user_query, k=7,fetch_k=20)

        # Merge retrieved chunks into one context block
        retrieved_context = "\n\n---\n\n".join([d.page_content for d in docs])



        response_placeholder.markdown("*Synthesizing complete response...*")

        # --- GENERATION PHASE ---
        system_instruction = (
            "You are an expert technical consultant for SAP BRIM (covering Convergent Mediation, "
            "Convergent Charging, Convergent Invoicing, and FI-CA).\n"
            "Your objective is to provide exhaustive, technically accurate answers using the provided manual context.\n"
            "Rules:\n"
            "1. Combine information across all provided document chunks into a coherent step-by-step guide.\n"
            "2. Include exact file paths, configuration parameters, and CLI commands mentioned in the text.\n"
            "3. Do not shorten or skip configuration steps.\n"
            "4. If the retrieved manual text genuinely lacks crucial details, explicitly mention what specific detail is missing.\n"
            "5. Do not answer questions outside of the SAP BRIM domain. If asked, politely inform the user to ask SAP BRIM questions only.\n"
            "6. Also if user query is related to SAP BRIM only and you dont have enough data from documentation then try to fetch data from extarnal sources as well but condition is only fetch data related to SAP BRIM."
        )

        prompt_payload = f"DOCUMENTATION CONTEXT:\n{retrieved_context}\n\nUSER QUESTION:\n{user_query}"


        def stream_groq_response(modelname):
            stream = client.chat.completions.create(
                model=modelname,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_payload}
                ],
                temperature=0.2,
                stream=True
            )
            full_answer = ""
            for chunk in stream:
                delta_text = chunk.choices[0].delta.content
                if delta_text:
                    full_answer += delta_text
                    response_placeholder.markdown(full_answer + "▌")
            return full_answer

        try:
            # 1st Attempt: Try primary model (70B)
            full_answer = stream_groq_response(model_Name)
        except RateLimitError:
            # 2nd Attempt: Automatically fall back to 8B if 70B hits daily quota (429)
            st.toast("70B limit reached. Automatically switching to Llama-3.1 8B!")
            try:
                full_answer = stream_groq_response(model_Name_bkp)
            except Exception as fallback_err:
                response_placeholder.markdown(" API Error: {str(fallback_err)}")
                full_answer = None

        except Exception as err:
            response_placeholder.markdown(" API Error: {str(err)}")
            full_answer = None

            # Add this right after generating full_answer:

            #unique_sources = list(set([
              #  f"📄 **{d.metadata.get('source', 'Manual')}** (Page {d.metadata.get('page', 'N/A')})"
              #  for d in docs
            #]))"""

            #"""citation_footer = "\n\n---\n📖 **Verified SAP BRIM Documentation References:**\n* " + "\n* ".join(
            #    unique_sources)"""

        if full_answer:
            response_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
