# Local Language Model Training

This project contains an implementation of a local language model based on DeepSeek Coder, optimized for both local and Google Colab training.

## Training Options

### 1. Google Colab (Recommended)
- Free GPU access
- Faster training
- No local installation required
- Follow instructions in `colab_instructions.txt`

### 2. Local Training
Only recommended if you have a good GPU (minimum 6GB VRAM)

## Hardware Requirements

### Minimum (CPU training):
- 8GB RAM (16GB recommended)
- Multi-core processor
- 20GB free disk space
- Windows with PowerShell
- Python 3.8+

### Recommended (GPU training):
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- CUDA-compatible GPU
- Windows with PowerShell
- Python 3.8+

## Installation Steps

1. **Create Virtual Environment**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install Dependencies**:
```powershell
python -m pip install --upgrade pip
pip install torch transformers datasets accelerate safetensors bitsandbytes peft
```

## Training Data Preparation

Place your training data in a text file (e.g., `training_data.txt`). Supported formats:
- Single .txt file
- Directory with multiple .txt files
- .jsonl file

Example data format:
```text
First example text.
Second example text.
...
```

## Training Commands

### For Google Colab:
```python
!python train_model_colab.py --data_path "training_data.txt" --epochs 3 --batch_size 1
```

### For Local CPU:
```powershell
python train_model.py --data_path "training_data.txt" --epochs 3 --batch_size 1
```

### For Local GPU:
```powershell
python train_model.py --data_path "training_data.txt" --epochs 3 --batch_size 4
```

## Training Parameters

1. `--epochs` (default: 3)
   - Number of training iterations
   - Higher values = better learning but risk of overfitting
   - Recommended: 3-5 for initial tests

2. `--batch_size` (default: 1)
   - Number of examples processed simultaneously
   - For 8GB RAM: use 1
   - For 16GB RAM: use 2
   - For GPU: use 4-8

3. `--data_path`
   - Path to your training data
   - Use quotes if path contains spaces
   - Example: `"C:\My Data\training_data.txt"`

## Model Configuration

Current optimized settings:
```python
# Memory Optimization
load_in_8bit=True
device_map="auto"
gradient_accumulation_steps=16

# Training Settings
learning_rate=1e-5
warmup_steps=50
max_grad_norm=1.0
```

## Project Structure
```
.
├── venv/                   # Python virtual environment
├── results/               # Training logs and checkpoints
├── offload/              # Temporary training files
├── local_language_model/ # Trained model output
├── requirements.txt      # Project dependencies
├── train_model.py       # Local training script
├── train_model_colab.py # Colab training script
├── test_model.py        # Model testing script
└── colab_instructions.txt # Colab setup guide
```

## Using the Trained Model

1. **Load Model**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./local_language_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```

2. **Generate Text**:
```python
prompt = "Your text here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_return_sequences=1,
    temperature=0.7
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce batch_size to 1
   - Enable 8-bit quantization
   - Use gradient accumulation

2. **CPU Memory Issues**:
   - Close other applications
   - Reduce batch_size
   - Use gradient checkpointing
   - Enable 8-bit quantization

3. **Training Too Slow**:
   - Use Google Colab (free GPU)
   - Reduce dataset size for testing
   - Reduce number of epochs

## Best Practices

1. **For Google Colab**:
   - Save model frequently
   - Download results after training
   - Use GPU runtime
   - Keep browser tab active

2. **For Local Training**:
   - Start with small dataset
   - Monitor RAM usage
   - Save checkpoints regularly
   - Use 8-bit quantization

## Model Features

- 8-bit quantization support
- LoRA adaptation capability
- Gradient checkpointing
- Automatic device mapping
- Memory optimization
- CPU and GPU support

## Notes

- Model uses 8-bit quantization by default
- Training progress is logged in results/
- Temporary files are stored in offload/
- Model is saved in local_language_model/
- All paths should use forward slashes or escaped backslashes

## Technical Notes

- Model uses half precision (fp16) for memory savings
- Uses gradient checkpointing for memory optimization
- Automatically detects and uses available GPU
- Supports offloading to CPU/disk when GPU memory is insufficient
- Default sequence length is 128 tokens

## Preparing Your Own Data

The project supports the following data formats:

1. Single text file (.txt)
2. Directory containing multiple .txt files
3. JSONL file (.jsonl) - each line contains JSON object with 'text' field

Example data structure:

```
data/
├── text1.txt         # Single text file
├── text2.txt
└── data.jsonl        # Or JSONL file with data
```

JSONL file format:
```jsonl
{"text": "First example text"}
{"text": "Second example text"}
```

## Testing the Model

To test the trained model, run in PowerShell:

```powershell
python test_model.py
```

## Differences Compared to GPT-2

- Better support for code generation
- More modern architecture
- Larger base model (1.3B parameters)
- Support for multiple programming languages
- Better understanding of technical context 

## Optimizing Training on a Weaker Machine

### Current Configuration for 8GB RAM:

1. **Tokenizer Optimization**:
   ```python
   max_length=32  # Reduced sequence length
   tokenizer.pad_token = tokenizer.eos_token  # Correct padding token configuration
   ```

2. **Model Optimization**:
   ```python
   model.config.use_cache = False  # Disable cache for memory savings
   model.config.pad_token_id = tokenizer.pad_token_id  # Correct padding handling
   torch_dtype=torch.float32  # Float32 precision for CPU
   low_cpu_mem_usage=True  # Memory optimization
   ```

3. **Training Parameters**:
   ```python
   gradient_accumulation_steps=16  # Increased gradient accumulation
   per_device_train_batch_size=1  # Minimum batch size
   gradient_checkpointing=True  # Memory savings at the cost of speed
   ```

4. **Additional Optimizations**:
   - Disable reporting (`report_to="none"`)
   - Disable pin memory (`dataloader_pin_memory=False`)
   - Group similar lengths (`group_by_length=True`)
   - No evaluation during training (`eval_strategy="no"`)

### Running Training on a Weak Machine:

1. **Environment Preparation**:
   - Close all other programs
   - Disable internet browsers
   - Check available memory in Task Manager

2. **Minimum Training Command**:
   ```powershell
   python train_model.py --data_path "training_data.txt" --epochs 1 --batch_size 1
   ```

3. **Monitoring**:
   - Monitor RAM usage in Task Manager
   - If RAM usage exceeds 90%, stop training
   - Check CPU temperature

4. **In Case of Memory Problems**:
   - Reduce training file size
   - Divide training into smaller parts
   - Increase `gradient_accumulation_steps`

### Expected Behavior:

1. **During Loading**:
   ```
   Loading tokenizer...
   Loading model...
   Loading training data...
   Tokenizing data...
   ```

2. **During Training**:
   - Slower training than on GPU
   - Possible high CPU usage
   - Regular progress updates

3. **After Training**:
   - Model saved in `local_language_model/`
   - Logs in `results/`
   - Temporary files in `offload/`

### CPU Training with Limited RAM

If you have a computer with limited RAM (8-16GB) and train on CPU, you can apply the following optimizations:

1. **Reducing Sequence Length**:
   ```python
   max_length=64  # Instead of default 128
   ```
   - Reduces memory usage by about 50%
   - Cost: model learns on shorter text fragments

2. **Increasing Gradient Accumulation**:
   ```python
   gradient_accumulation_steps=8  # Instead of default 4
   ```
   - Allows training with larger effective batch size without increasing memory usage
   - Cost: slower training

3. **Learning Parameter Optimization**:
   ```python
   learning_rate=1e-5  # Instead of 2e-5
   warmup_steps=50    # Instead of 100
   ```
   - Stable training on CPU
   - Lower memory overflow risk

4. **Removing Unnecessary Functions**:
   ```python
   load_best_model_at_end=False
   evaluation_strategy="no"
   save_total_limit=1
   ```
   - Saves memory by not evaluating and storing multiple model copies
   - Cost: no ability to choose the best model

5. **Memory Management**:
   ```python
   device_map="auto"
   low_cpu_mem_usage=True
   ```
   - Automatic memory management
   - Better use of available resources

### Recommended Settings for Different Hardware Configurations:

#### 8GB RAM (minimum):
```powershell
python train_model.py --data_path "training_data.txt" --epochs 1 --batch_size 1
```
- Use small training files
- Close all other programs
- Prepare for long training time

#### 16GB RAM:
```powershell
python train_model.py --data_path "training_data.txt" --epochs 2 --batch_size 1
```
- You can use larger training files
- You can increase the number of epochs
- Still recommended to close other programs

#### 32GB RAM:
```powershell
python train_model.py --data_path "training_data.txt" --epochs 3 --batch_size 2
```
- You can use default settings
- Greater freedom in training data size

### Monitoring Resource Usage:

1. **Windows Task Manager**:
   - Ctrl + Shift + Esc
   - "Performance" tab
   - Monitor RAM and CPU usage

2. **Warning Indicators**:
   - If RAM usage exceeds 90%
   - If CPU temperature is high
   - If system slows down

3. **What to Do in Case of Problems**:
   - Reduce `batch_size` to 1
   - Reduce `max_length`
   - Increase `gradient_accumulation_steps`
   - In the end, reduce model size 