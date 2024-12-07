import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import bitsandbytes as bnb
import evaluate

# Define OPTQ Quantization Helpers
def optq_quantize(weights, num_bits=8):
    """OPTQ-style quantization for weight matrices."""
    scale = weights.abs().max() / (2**(num_bits - 1) - 1)
    quantized = (weights / scale).round().to(torch.int8)
    return quantized, scale

def optq_dequantize(quantized, scale):
    """Dequantize the quantized weights."""
    return quantized.float() * scale

# ModularQLoRALayer Class
class ModularQLoRALayer(nn.Module):
    def __init__(self, base_layer, rank, num_bits=8):
        super().__init__()
        self.base_layer = base_layer
        # Quantize weights with OPTQ
        self.quantized_weights, self.scale = optq_quantize(base_layer.weight, num_bits)
        # Get input and output dimensions
        input_dim = base_layer.in_features
        output_dim = base_layer.out_features
        # Initialize low-rank LoRA matrices
        self.A = nn.Parameter(torch.randn(rank, output_dim) * 0.01)  # Shape: (rank, output_dim)
        self.B = nn.Parameter(torch.randn(input_dim, rank) * 0.01)  # Shape: (input_dim, rank)
        self.bias = base_layer.bias  # Copy bias if present

    def forward(self, x):
        # Move quantized weights to the same device as the input
        dequantized_weights = optq_dequantize(self.quantized_weights, self.scale).to(x.device)
        self.A = self.A.to(x.device)
        self.B = self.B.to(x.device)

        # Compute LoRA adaptation
        lora_output = (x @ self.B) @ self.A

        # Combine dequantized weights and LoRA output
        output = x @ dequantized_weights.t() + lora_output
        if self.bias is not None:
            self.bias = self.bias.to(x.device)
            output += self.bias
        return output

# Replace Layers in the Model
class ModularQLoRAAdapter(nn.Module):
    def __init__(self, model, rank, num_bits=8):
        super().__init__()
        self.model = model
        # Copy the modules to avoid dictionary size errors during iteration
        modules_copy = list(model.named_modules())
        for name, module in modules_copy:
            if isinstance(module, nn.Linear):
                # Get the parent module if the layer is nested
                parent_name, child_name = name.rsplit(".", 1) if "." in name else (None, name)
                # Replace nn.Linear with ModularQLoRALayer
                new_module = ModularQLoRALayer(module, rank=rank, num_bits=num_bits)
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, new_module)
                else:
                    setattr(model, name, new_module)

# Load the Model and Tokenizer
model_name = "Helsinki-NLP/opus-mt-mul-en"  # Pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def count_parameters(model):
    """Returns the number of trainable and total parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

# Freeze all non-LoRA parameters
for param in base_model.parameters():
    param.requires_grad = False

# Add ModularQLoRA Layers
rank = 4
num_bits = 8
model = ModularQLoRAAdapter(base_model, rank=rank, num_bits=num_bits).model
# Count trainable and total parameters
trainable_params, total_params = count_parameters(model)
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Total Parameters: {total_params:,}")
print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")

# Define the Dataset
dataset = load_dataset("wmt14", "de-en", split="train[:1%]")  # Subset for testing
eval_dataset = load_dataset("wmt14", "de-en", split="test")  # Newstest2014 for evaluation



# Preprocessing Function
def preprocess_function(examples):
    inputs = [ex["de"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=128, truncation=True, padding=True).input_ids
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in l] for l in labels]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_train_dataset = dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", label_pad_token_id=-100)

# Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results-modular-q-lora",
    evaluation_strategy="no",  # Skip intermediate evaluation
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    fp16=torch.cuda.is_available(),  # Use mixed precision if possible
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_safetensors=False,
)

# Metric for Evaluation
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]
    return bleu.compute(predictions=decoded_preds, references=decoded_labels)

# Use 8-bit Optimizer
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

# Trainer Setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),  # Use 8-bit optimizer
)

# Train the Model
print("Starting training...")
trainer.train()

# Evaluate on Newstest2014 Dataset
print("Evaluating on newstest2014...")
metrics = trainer.evaluate(eval_dataset=tokenized_eval_dataset)
print(metrics)