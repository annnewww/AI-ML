import os
import shutil
import time
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Calculate absolute paths based on the location of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "data")
DB_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "db")


def clear_old_database(db_path):
    """Deletes the existing ChromaDB directory cleanly."""
    if os.path.exists(db_path):
        print("Clearing old database at: {db_path}")
        try:
            shutil.rmtree(db_path)
            time.sleep(1)
            print("Old database deleted successfully.")
        except Exception as e:
            print("Warning: Could not delete DB directly ({e}). Trying to remove contents...")
            for root, dirs, files in os.walk(db_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))


def build_vector_database():
    # Step 0: Always delete old DB first
    clear_old_database(DB_PATH)

    print("\nStep 1: Loading raw documents from data/ folder...")

    # Fast PyMuPDFLoader for PDFs
    pdf_loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )

    documents = pdf_loader.load()

    if not documents:
        print("Error: No PDF documents found in '{DATA_PATH}'. Please check your files!")
        return

    print("Loaded {len(documents)} raw pages across all PDFs.")

    print("\nStep 2: Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        #chunk_size=1200,
        chunk_size=2000,
        #chunk_overlap=200,
        chunk_overlap=300,
        separators=[
            "\n## ",      # Markdown H2
            "\n### ",     # Markdown H3
            "\n\n",       # Paragraph breaks
            "\n- ",       # Bullet points
            "\n1. ",      # Numbered lists
            "\n",         # Line breaks
            " ",
            ""
        ]
    )
    chunks = text_splitter.split_documents(documents)
    print("Generated {len(chunks)} individual text chunks.")

    print("\nStep 3: Initializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 64}
    )

    print("\nStep 4: Saving new chunks to local Vector DB (Chroma)...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

    print("Success! Fresh vector database built and saved to: {DB_PATH}")


if __name__ == "__main__":
    # Ensure data folder exists
    os.makedirs(DATA_PATH, exist_ok=True)
    build_vector_database()