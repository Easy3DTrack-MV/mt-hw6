import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import evaluate
import time
import psutil  # For CPU memory profiling
import os

# Define the LoRA Layer
class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank):
        super().__init__()
        self.base_layer = base_layer  # Original layer
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        self.scaling = 1 / rank  # Scale down LoRA's contribution

    def forward(self, x):
        # Add the LoRA adaptation to the base layer's output
        return self.base_layer(x) + self.scaling * self.lora_B(self.lora_A(x))


# Function to Add LoRA Adapters
def add_lora_adapters(model, rank):
    """
    Adds LoRA adapters to the model without modifying the dictionary during iteration.
    """
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_replace.append(name)

    for module_name in modules_to_replace:
        if '.' in module_name:  # Nested module
            parent_name, child_name = module_name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
            original_module = getattr(parent_module, child_name)
            setattr(parent_module, child_name, LoRALayer(original_module, rank))
        else:  # Top-level module
            original_module = getattr(model, module_name)
            setattr(model, module_name, LoRALayer(original_module, rank))

    return model


# Save LoRA Adapters
def save_lora_adapters(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight
    torch.save(lora_state_dict, os.path.join(save_dir, "lora_adapters.pth"))


# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-mul-en"  # mul-to-English model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Add LoRA adapters with a specified rank
rank = 4
model = add_lora_adapters(model, rank)

# Ensure only LoRA parameters are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable Parameters: {trainable_params} / {total_params}")

# Load dataset (WMT14 English-German)
dataset = load_dataset("wmt14", "de-en", split="train[:1%]")  # Subset for testing

# Preprocess dataset
def preprocess_function(examples):
    inputs = [ex["de"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"].select(range(500))

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results-lora",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=10,
    save_safetensors=False,
    gradient_checkpointing=True,
)

# Metric for evaluation
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]
    return bleu.compute(predictions=decoded_preds, references=decoded_labels)

# Use DataCollator for processing
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Memory Profiling Function
def log_memory_usage(step):
    gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    cpu_memory = psutil.virtual_memory().used / 1024**2
    print(f"Step {step}: GPU Memory Usage: {gpu_memory:.2f} MB, CPU Memory Usage: {cpu_memory:.2f} MB")

# Training with Memory Profiling
start_time = time.time()
print("Starting training...")
for epoch in range(training_args.num_train_epochs):
    print(f"\nEpoch {epoch + 1}/{training_args.num_train_epochs}")
    trainer.train()
    log_memory_usage(f"Epoch {epoch + 1} (After Training)")

end_time = time.time()

# Save only the LoRA adapters
save_lora_adapters(model, "./results-lora-adapters")

# Summary
print("\nSummary:")
print(f"Training Time: {end_time - start_time:.2f} seconds")
log_memory_usage("Final")