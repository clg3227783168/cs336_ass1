# CS336 Assignment 1 - Implementing a Transformer Language Model from Scratch

[中文版本](README.md)

This project implements a Transformer-based language model from scratch according to the requirements of CS336 Assignment 1. The project includes a complete model architecture, tokenizer, training and evaluation pipeline, and is trained on the TinyStories dataset.

## Project Overview

- **Goal**: Implement a basic Transformer language model to understand its core components and working principles
- **Key Features**:
  - Transformer-based language model
  - BPE (Byte-Pair Encoding) tokenizer
  - Model training and evaluation pipeline
  - Text generation functionality

## Model Architecture

This project implements a standard Transformer decoder model with the following characteristics:

- **Vocabulary size**: 32000
- **Context length**: 256
- **Model dimension**: 512
- **Number of layers**: 4
- **Attention heads**: 16
- **Feed-forward network dimension**: 1344
- **Positional encoding**: RoPE (Rotary Position Embedding)

The model includes the following core components:
- Multi-head self-attention mechanism
- Feed-forward neural network
- Residual connections and layer normalization
- Causal attention mask (ensuring the model can only see previous tokens)

## File Structure

- `basics/`: Source code directory containing all basic code, with functional modules implemented as classes
  - `transformer.py`: Core implementation of the Transformer model
  - `tokenizer.py`: BPE tokenizer implementation
  - `trainer_model.py`: Model training pipeline
  - `trainer_tokenizer.py`: Tokenizer training pipeline
  - `eva_pretrain.py`: Pretrained model evaluation
  - `nn_utils.py` & `trainer_utils.py`: Utility functions
- `tests/`: Test code directory, with adapters.py implementing all functionality through functions
- `scripts/`: Script directory
  - `pretrain_model.py`: Pretrain model script
  - `train_tokenizer.py`: Train tokenizer script
  - `eva_pretrain_model.py`: Evaluate pretrained model script
  - `configs/`: Configuration file directory
- `data/`: Data directory for storing the TinyStories dataset

## Dataset

This project uses the TinyStories dataset for training. TinyStories is a dataset composed of simple English short stories designed specifically for training small language models. The dataset includes:
- Training set: `TinyStoriesV2-GPT4-train.dat`
- Validation set: `TinyStoriesV2-GPT4-valid.dat`

Original dataset download links:
- Training set: [TinyStoriesV2-GPT4-train.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt)
- Validation set: [TinyStoriesV2-GPT4-valid.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt)

## Environment Setup and Running

### Dependency Management

This project uses uv for dependency management, which provides faster package installation and environment management compared to traditional pip and conda.

### Running Guide

```bash
# Train tokenizer
uv run scripts/train_tokenizer.py

# Pretrain model
uv run scripts/pretrain_model.py

# Evaluate pretrained model
uv run scripts/eva_pretrain_model.py
```

## Pretraining

- **Hardware**: Tesla T4 GPU
- **Dataset**: TinyStories
- **Training duration**: 30 minutes
- **Training configuration**:
  - Learning rate: 0.0005
  - Minimum learning rate: 0.0001
  - Weight decay: 0.01
  - Batch size: 32
  - Context length: 256
  - Training steps: 5000
  - Gradient clipping: 1.0
  - Learning rate schedule: warmup (500 steps) + cosine decay (5000 steps)

## Training Results

The following figure shows the loss curve during training:

![Training Loss Curve](assets/loss.png)

From the loss curve, we can see that the model's loss steadily decreases during training, indicating good learning performance.

## Text Generation Example

After training, the model can be used to generate text. Here is example code for generating text using the model:

```python
from basics.transformer import Transformer
from basics.tokenizer import Tokenizer

# Load model and tokenizer
model = Transformer.from_pretrained("checkpoints/cs336_lm_TinyStories")
tokenizer = Tokenizer.from_pretrained("checkpoints/cs336_lm_TinyStories")

# Generate text
prompt = "Once upon a time"
generated_text = model.generate(
    tokenizer.encode(prompt),
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
)
print(tokenizer.decode(generated_text))
```