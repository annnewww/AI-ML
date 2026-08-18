import os
import pickle
import shutil
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "data")
DB_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "db")
CHUNKS_PATH = os.path.join(BASE_DIR, "mediation-ai-tool", "chunks")

def clear_old_database(db_path):
    """Deletes the existing ChromaDB directory cleanly before rebuilding."""
    if os.path.exists(db_path):
        print(f"Clearing old database at: {db_path}")
        try:
            shutil.rmtree(db_path)
            time.sleep(1)
            print("Old database deleted successfully.")
        except Exception as e:
            print(f"Warning: Could not delete DB directly ({e}). Cleaning contents...")
            for root, dirs, files in os.walk(db_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))


def build_vector_database():
    # 0. Always delete old DB first to prevent stale chunk mixing
    clear_old_database(DB_PATH)

    print("\nStep 1: Loading raw documents & injecting metadata...")
    documents = []

    # Traverse subfolders inside DATA_PATH (e.g., data/BRIM, data/RAR, data/FSCM, data/FI)
    for root, dirs, files in os.walk(DATA_PATH):
        # Infer module name from folder name
        folder_name = os.path.basename(root)

        # Skip the root DATA_PATH directory if files are inside subfolders
        if root == DATA_PATH:
            continue

        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"  --> Processing [{folder_name.upper()}]: {file}")

                loader = PyMuPDFLoader(pdf_path)
                loaded_pages = loader.load()

                # --- STEP 3 METADATA INJECTION ---
                for page in loaded_pages:
                    # Explicitly attach module and file name to metadata dictionary
                    page.metadata["module"] = folder_name.upper()
                    page.metadata["source_file"] = file

                documents.extend(loaded_pages)

    if not documents:
        print(f"❌ Error: No PDF files found in subfolders under '{DATA_PATH}'.")
        print("Please ensure your folder structure looks like: data/BRIM/, data/RAR/, etc.")
        return

    print(f"\nSuccessfully loaded {len(documents)} total pages across all manuals.")

    print("\nStep 2: Chunking documents (Preserving Metadata)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=[
            "\n## ",  # Markdown H2
            "\n### ",  # Markdown H3
            "\n\n",  # Paragraph breaks
            "\n- ",  # Bullet points
            "\n1. ",  # Numbered lists
            "\n",  # Line breaks
            " ",
            ""
        ]
    )

    # Split documents into 2000-char chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} metadata-tagged chunks.")

    # Save raw chunks to pickle file for BM25
    print("Saving chunk metadata to chunks.pkl for BM25...")
    pickle_path = os.path.join(CHUNKS_PATH, "chunks.pkl")
    with open(pickle_path, "wb") as data:
        pickle.dump(chunks, data)

    # Quick sanity check print on first chunk
    if chunks:
        print("\n--- [Sanity Check: Chunk 1 Metadata] ---")
        print(f"Module Tag  : {chunks[0].metadata.get('module')}")
        print(f"Source File : {chunks[0].metadata.get('source_file')}")
        print("------------------------------------------")

    print("\nStep 3: Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 64}
    )

    print("\nStep 4: Saving tagged chunks to ChromaDB...")
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    print(f"\n✅ SUCCESS! Vector database freshly indexed at: {DB_PATH}")


if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    build_vector_database()