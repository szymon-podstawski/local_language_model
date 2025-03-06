from datasets import Dataset
import os
import json

def load_text_files(data_dir):
    """Loads text files from directory."""
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                texts.append({'text': f.read()})
    return texts

def load_jsonl_file(file_path):
    """Loads data from JSONL file."""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data, dict) and 'text' in data:
                texts.append(data)
            elif isinstance(data, str):
                texts.append({'text': data})
    return texts

def create_dataset(data_path):
    """
    Creates dataset from custom data.
    
    Args:
        data_path: Path to .txt/.jsonl file or directory with .txt files
    
    Returns:
        Dataset: Dataset ready for training
    """
    if os.path.isdir(data_path):
        # Load all .txt files from directory
        texts = load_text_files(data_path)
    elif data_path.endswith('.jsonl'):
        # Load data from JSONL file
        texts = load_jsonl_file(data_path)
    elif data_path.endswith('.txt'):
        # Load single text file
        texts = [{'text': open(data_path, 'r', encoding='utf-8').read()}]
    else:
        raise ValueError("Unsupported data format. Use .txt, .jsonl or directory with .txt files")

    # Convert to datasets format
    dataset = Dataset.from_list(texts)
    
    print(f"Loaded {len(dataset)} examples")
    return dataset 