from datasets import load_dataset
import random

def preview_wikitext_data(num_samples=5):
    """Display sample data from WikiText-2"""
    print("=== Preview of WikiText-2 Data ===\n")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Display basic information
    print(f"Total number of examples: {len(dataset)}")
    print(f"Data structure example: {dataset.features}\n")
    
    # Display random examples
    print(f"=== {num_samples} random examples ===\n")
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in random_indices:
        print(f"Example #{idx}:")
        print("=" * 50)
        print(dataset[idx]['text'])
        print("=" * 50)
        print("\n")

def preview_custom_data(file_path):
    """Display sample data from custom text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
            
        print(f"\n=== Preview of custom data from {file_path} ===\n")
        print(f"Total number of lines: {len(content)}")
        
        # Display first 5 lines
        print("\nFirst 5 lines:")
        print("=" * 50)
        for line in content[:5]:
            print(line.strip())
        print("=" * 50)
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")

if __name__ == "__main__":
    # Preview data from WikiText-2
    preview_wikitext_data()
    
    # Example usage for custom data
    # Uncomment the line below and provide path to your file
    # preview_custom_data("path/to/your/data.txt") 