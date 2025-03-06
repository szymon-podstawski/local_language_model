from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Load trained model and tokenizer
    model_path = "./local_language_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Example text to complete
    prompt = "In today's world, artificial intelligence"

    # Generate text
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )

    # Decode and display generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main() 