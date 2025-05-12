import chromadb
import numpy as np
from tqdm import tqdm
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chromadb_test.log'),
        logging.StreamHandler()
    ]
)

# Connect to ChromaDB with proper error handling
try:
    logging.info(f"Attempting to connect to ChromaDB at 34.44.238.198:8000")
    client = chromadb.HttpClient(host="34.44.238.198", port=8000)
    
    # Test connection with a simple operation
    collections = client.list_collections()
    logging.info(f"Connection successful. Found {len(collections)} collections.")
except Exception as e:
    logging.error(f"Failed to connect to ChromaDB: {str(e)}")
    sys.exit(1)

# Create a test collection
try:
    collection_name = "test_collection"
    collection = client.get_or_create_collection(collection_name)
    logging.info(f"Created/accessed collection: {collection_name}")
    
    # Generate some random test data
    num_items = 1000
    dimension = 384  # Common embedding dimension
    
    embeddings = np.random.rand(num_items, dimension)
    documents = [f"This is test document {i}" for i in range(num_items)]
    ids = [str(i) for i in range(num_items)]
    
    # Insert data in batches
    batch_size = 100
    num_batches = len(ids) // batch_size
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_embeddings = embeddings[start_idx:end_idx].tolist()
        batch_documents = documents[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_documents,
            ids=batch_ids
        )
        logging.info(f"Inserted batch {i+1}/{num_batches}")
    
    logging.info(f"Successfully inserted {num_items} items into ChromaDB")
except Exception as e:
    logging.error(f"Error during ChromaDB operations: {str(e)}")
    sys.exit(1)
