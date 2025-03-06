from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_path):
    """Loads the trained model and tokenizer."""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=100):
    """Generates text based on the given prompt."""
    # Input preparation
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generation
    print(f"\nGenerating text for prompt: {prompt}")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decoding and displaying result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    # Path to the unpacked model
    model_path = "./local_language_model"
    
    # Loading the model
    model, tokenizer = load_model(model_path)
    
    # Example prompts for testing
    test_prompts = [
        "Python is a programming language that",
        "A function in Python is used to",
        "To create a new class in Python, you need to"
    ]
    
    # Testing the model
    for prompt in test_prompts:
        generated = generate_text(prompt, model, tokenizer)
        print(f"\nResult:\n{generated}\n")
        print("-" * 50)
        
    # Interactive mode
    print("\nType 'quit' to exit.")
    while True:
        user_prompt = input("\nEnter text to complete: ")
        if user_prompt.lower() == 'quit':
            break
            
        generated = generate_text(user_prompt, model, tokenizer)
        print(f"\nResult:\n{generated}")

if __name__ == "__main__":
    main() 