from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import argparse
from custom_dataset import create_dataset
import os

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training model on custom data')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to .txt/.jsonl file or data directory')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size')
    args = parser.parse_args()

    # Creating directories for the model
    os.makedirs("offload", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load DeepSeek model
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    # Setting padding token
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    # Loading model in CPU mode with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        offload_folder="offload"
    )

    # Model configuration
    model.config.use_cache = False  # Disable cache for gradient checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to("cpu")

    # Load custom data
    print("Loading training data...")
    dataset = create_dataset(args.data_path)

    def tokenize_function(examples):
        # Text tokenization
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=32,
            return_tensors=None
        )
        
        # Add labels (same as input_ids)
        encodings["labels"] = encodings["input_ids"].copy()
        
        return encodings

    # Prepare data
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=1000,
        save_total_limit=1,
        fp16=False,
        gradient_accumulation_steps=16,
        logging_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        use_cpu=True,  # Using new parameter instead of no_cuda
        load_best_model_at_end=False,
        save_strategy="steps",
        eval_strategy="no",  # Using new parameter instead of evaluation_strategy
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        group_by_length=True
    )

    # Trainer initialization
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model
    print("Saving model...")
    model.save_pretrained("./local_language_model")
    tokenizer.save_pretrained("./local_language_model")
    print("Training completed!")

if __name__ == "__main__":
    main()