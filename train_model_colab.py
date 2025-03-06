from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import argparse
from datasets import Dataset
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def create_dataset(file_path):
    """Creates dataset from text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    print(f"Loaded {len(texts)} lines of text")
    return Dataset.from_dict({"text": texts})

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir="results/tensorboard")

    def training_step(self, model, inputs, return_loss=True, num_items_in_batch=None):
        """Performs one training step."""
        if return_loss:
            loss = super().training_step(model, inputs, return_loss)
            self.writer.add_scalar('train/loss', loss.item(), self.state.global_step)
            print(f"Step {self.state.global_step}: loss = {loss.item():.4f}")
            return loss
        return super().training_step(model, inputs, return_loss)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model to {output_dir}")
        super().save_model(output_dir, _internal_call)

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training model on custom data')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to .txt/.jsonl file or data directory')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    args = parser.parse_args()

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Clean and create directories
    if os.path.exists("results"):
        print("Removing old results directory...")
        shutil.rmtree("results")
    if os.path.exists("local_language_model"):
        print("Removing old local_language_model directory...")
        shutil.rmtree("local_language_model")

    print("Creating new directories...")
    os.makedirs("results/tensorboard", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("local_language_model", exist_ok=True)

    # Load DeepSeek model
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA Configuration
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Add LoRA adapter
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        print(f"Model loaded successfully in 8-bit mode with LoRA")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    # Model configuration
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load custom data
    print("Loading training data...")
    dataset = create_dataset(args.data_path)

    def tokenize_function(examples):
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors=None
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    # Prepare data
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        logging_steps=1,
        learning_rate=1e-4,  # Increased learning rate for LoRA
        warmup_steps=50,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        save_strategy="epoch",
        report_to=["tensorboard"],
        group_by_length=True,
        logging_dir="results/tensorboard",
        load_best_model_at_end=False,
        evaluation_strategy="no"
    )

    # Ensure all parameters are in float32
    for param in model.parameters():
        param.data = param.data.to(torch.float32)

    print(f"Training configuration:")
    print(f"- Precision: 8-bit with LoRA")
    print(f"- Learning rate: {training_args.learning_rate}")
    print(f"- Batch size: {training_args.per_device_train_batch_size}")
    print(f"- Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"- Max grad norm: {training_args.max_grad_norm}")
    print(f"- Device: {device.type}")

    # Initialize trainer
    print("Initializing trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

    # Save model
    print("Saving model...")
    try:
        model_path = "./local_language_model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved in {model_path}")
        
        # Check if model was saved
        if not os.path.exists(model_path):
            raise Exception("Model directory was not created")
        
        # Try to create ZIP archive
        if os.system(f"zip -r local_language_model.zip {model_path}") != 0:
            raise Exception("Failed to create ZIP archive")
            
        print("Model has been successfully saved and packed!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

if __name__ == "__main__":
    main() 