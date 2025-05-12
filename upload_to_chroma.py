import chromadb
import argparse
from utils_map import info_map
import io
from google.cloud import storage
import json
from tqdm import tqdm as tqdm
import numpy as np
from emb_list import emb_file_names
import logging
import os
import datetime

# Setup logging
def setup_logging(log_dir="logs"):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"upload_to_chroma_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def load_jsonl(bucket_name:str,file_no:int)->list[dict]:
    file_name = f'datasets/datasets--chentong00--factoid-wiki/snapshots/60bce4923950eab87192e276c9c5e5136234a760/data/docs-{file_no:04d}_of_1000.jsonl'
    
    logging.info(f"Loading JSONL file {file_no:04d} from bucket {bucket_name}")
    
    file_obj, file_name = load_file_from_gcs(bucket_name,file_path = file_name)

    text = file_obj.getvalue().decode("utf-8")

    data_list = [json.loads(line) for line in text.splitlines()]
    
    logging.info(f"Loaded {len(data_list)} records from file {file_no:04d}")

    return data_list 

def load_array(bucket_name:str,embedding_file):
    logging.info(f"Loading embedding arrays from {embedding_file[0]} to {embedding_file[1]}")

    file_name_template = 'embeddings/embeddings_{start_id}-{end_id}.npy'

    start_file = file_name_template.format(start_id = embedding_file[0][0], end_id = embedding_file[0][1])
    end_file = file_name_template.format(start_id = embedding_file[1][0], end_id = embedding_file[1][1])

    start_flag = False
    for file_name in emb_file_names:

        if file_name == start_file:
            start_flag = True

        if start_flag:
            logging.info(f"Loading embedding file: {file_name}")
            file_obj, file_name = load_file_from_gcs(bucket_name,file_name)
            data = np.load(file_obj)
            logging.info(f"Loaded embedding array with shape {data.shape}")

            yield data
        
        if file_name == end_file:
            break
    

def list_blobs(bucket_name, suffix=None):
    """Lists all the blobs in the bucket with optional suffix filtering.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        suffix (str or list): File suffix(es) to filter by (e.g. '.jsonl', '.npy')
        
    Returns:
        list: List of blob objects matching the suffix criteria
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs()
    
    if suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        filtered_blobs = [blob.name for blob in blobs if any(blob.name.endswith(s) for s in suffix)]
        return filtered_blobs
    
    return list(blobs)

def load_file_from_gcs(bucket_name, file_path):
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(file_path)

    if not blob.exists():
        logging.error(f"File not found: gs://{bucket_name}/{file_path}")
        raise FileNotFoundError(f"File not found: gs://{bucket_name}/{file_path}")

    logging.debug(f"Loading file from GCS: gs://{bucket_name}/{file_path}")

    # Download the file as bytes into memory
    data_bytes = blob.download_as_bytes()
    file_obj = io.BytesIO(data_bytes)
    
    logging.debug(f"Successfully loaded file: {blob.name} ({len(data_bytes)} bytes)")

    return file_obj, blob.name

def load_prop_data(bucket_name, start_text_file_num,end_text_file_num,start_text_file_line_no,end_text_file_line_no):
    logging.info(f"Loading proposition data from file {start_text_file_num} (line {start_text_file_line_no}) to file {end_text_file_num} (line {end_text_file_line_no})")
    
    current_text_file_num = start_text_file_num
    current_text_file_line_no = start_text_file_line_no 

    doc_json_data = load_jsonl(bucket_name,current_text_file_num)

    for data in doc_json_data[current_text_file_line_no:]:
        yield data['contents'] 

    current_text_file_num += 1


    while current_text_file_num < end_text_file_num:
        logging.info(f"Processing file {current_text_file_num}")
        doc_json_data = load_jsonl(bucket_name,current_text_file_num)

        for data in doc_json_data:
            yield data['contents'] 

        current_text_file_num += 1

    del doc_json_data

    logging.info(f"Processing final file {end_text_file_num}")
    last_doc_json_data = load_jsonl(bucket_name,end_text_file_num)

    for data in last_doc_json_data[:end_text_file_line_no+1]:
        yield data['contents'] 

    del last_doc_json_data
    logging.info("Finished loading all proposition data")

if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Upload data to ChromaDB')
    parser.add_argument('--host', type=str, default='localhost', help='ChromaDB host address')
    parser.add_argument('--port', type=int, default=8000, help='ChromaDB port number')
    parser.add_argument('--part-number', type=str, required=True, help='Part number to process')

    args = parser.parse_args()
    host = f"{args.host}"
    
    logging.info(f"Starting upload to ChromaDB with parameters: host={args.host}, port={args.port}, part-number={args.part_number}")

    try:
        client = chromadb.HttpClient(host = host, port = args.port)
        client.heartbeat()
        logging.info("Successfully connected to ChromaDB")

        collection = client.get_or_create_collection("props")
        logging.info("Using collection: props")

        part_info = info_map[args.part_number] 
        logging.info(f"Processing part {args.part_number} with info: {part_info}")

        start_embedding_file = part_info['embedding_files'][0]
        end_embedding_file = part_info['embedding_files'][1]

        start_text_file_num = part_info['text_files'][0][0]
        end_text_file_num = part_info['text_files'][1][0]

        start_text_file_line_no = part_info['text_files'][0][1] 
        end_text_file_line_no = part_info['text_files'][1][1]

        bucket_name = "proposition-vectors"
        logging.info(f"Using GCS bucket: {bucket_name}")

        prop_generator = load_prop_data(bucket_name,start_text_file_num,end_text_file_num,start_text_file_line_no,end_text_file_line_no)

        global_counter = 0

        batch_size = 2500
        logging.info(f"Using batch size: {batch_size}")

        batch_vectors = []
        batch_ids = []
        batch_documents = []

        batch_counter = 0
        for embedding_file in tqdm(load_array(bucket_name,[start_embedding_file,end_embedding_file])):
            logging.info(f"Processing embedding file with {len(embedding_file)} vectors")
            
            for vector in embedding_file:
                prop = next(prop_generator)
                batch_vectors.append(vector)
                batch_ids.append(str(global_counter))
                batch_documents.append(prop)
                global_counter += 1
                batch_counter += 1

                if batch_counter == batch_size:
                    logging.info(f"Uploading batch of {batch_size} items to ChromaDB (total processed: {global_counter})")
                    collection.add(
                        ids = batch_ids,
                        embeddings = batch_vectors,
                        documents = batch_documents
                    )

                    batch_vectors = []
                    batch_ids = []
                    batch_documents = []
                    batch_counter = 0

        if batch_counter > 0:
            logging.info(f"Uploading final batch of {batch_counter} items to ChromaDB (total processed: {global_counter})")
            collection.add(
                ids = batch_ids,
                embeddings = batch_vectors,
                documents = batch_documents
            )
        
        logging.info(f"Upload completed successfully. Total records processed: {global_counter}")
        logging.info(f"Log saved to: {log_file}")
    
    except Exception as e:
        logging.error(f"Error during upload process: {str(e)}", exc_info=True)
        raise
    

