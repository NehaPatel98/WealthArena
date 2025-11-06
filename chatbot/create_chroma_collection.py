"""
WealthArena - Chroma DB Collection Creator
Loads financial embeddings from chatbot_setup into Chroma vector database

Usage:
    python create_chroma_collection.py

Prerequisites:
    - Run 02_create_embeddings.py first to generate financial_embeddings.json
    - chromadb package installed (pip install chromadb==0.4.22)
"""

import json
import os
from pathlib import Path

# Try to import Chroma
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    print("ERROR: chromadb package not installed")
    print("Install it with: pip install chromadb==0.4.22")
    exit(1)

def load_embeddings(file_path: str) -> dict:
    """Load embeddings from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Embeddings file not found: {file_path}")
        print("Please run 02_create_embeddings.py first to generate financial_embeddings.json")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in embeddings file: {e}")
        exit(1)

def create_chroma_collection():
    """Create Chroma collection and load embeddings"""
    # Path to embeddings file
    embeddings_file = Path(__file__).parent.parent / "chatbot_setup" / "financial_embeddings.json"
    
    # Load embeddings
    print(f"Loading embeddings from: {embeddings_file}")
    embeddings_data = load_embeddings(str(embeddings_file))
    
    print(f"Loaded {len(embeddings_data)} terms")
    
    # Initialize Chroma client
    chroma_dir = Path(__file__).parent / "data" / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing Chroma client at: {chroma_dir}")
    client = chromadb.Client(Settings(
        persist_directory=str(chroma_dir),
        anonymized_telemetry=False
    ))
    
    # Create or get collection
    collection_name = "financial_knowledge"
    print(f"Creating collection: {collection_name}")
    
    try:
        # Try to get existing collection first
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists. Clearing existing data...")
        # Chroma doesn't have a direct clear method, so we'll delete and recreate
        client.delete_collection(name=collection_name)
    except:
        pass  # Collection doesn't exist, which is fine
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare data for Chroma
    ids = []
    embeddings_list = []
    metadatas = []
    documents = []
    
    print("Preparing data for Chroma...")
    for term_id, data in embeddings_data.items():
        ids.append(term_id)
        embeddings_list.append(data['embedding'])
        metadatas.append({
            'term': data['term'],
            'definition': data['definition'],
            'url': data.get('url', ''),
            'source': data.get('source', 'investopedia')
        })
        documents.append(data['text'])
    
    # Add to collection in batches to handle large datasets
    batch_size = 100
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"Adding {len(ids)} items to Chroma in {total_batches} batches...")
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = embeddings_list[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_documents = documents[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
        
        batch_num = (i // batch_size) + 1
        print(f"  Batch {batch_num}/{total_batches} added ({len(batch_ids)} items)")
    
    # Verify collection
    count = collection.count()
    print(f"\n✅ Successfully loaded {count} items into Chroma collection '{collection_name}'")
    print(f"📁 Chroma database persisted at: {chroma_dir}")
    print(f"\n🎯 Collection is ready for use by the chatbot search endpoint!")
    print(f"   The search endpoint will automatically use Chroma if available.")

if __name__ == "__main__":
    print("=" * 60)
    print("WealthArena - Chroma DB Collection Creator")
    print("=" * 60)
    print()
    
    create_chroma_collection()
    
    print()
    print("=" * 60)
    print("✅ Chroma collection creation completed!")
    print("=" * 60)

