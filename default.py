import torch
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

# Load model and tokenizer
# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"  # Pretrained en-fr model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Freeze all parameters except specific layers
for name, param in model.named_parameters():
    if not name.startswith("model.decoder.layers.5") and not name.startswith("model.decoder.layers.6") and "lm_head" not in name:
        param.requires_grad = False

# Verify trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable Parameters: {trainable_params} / {total_params}")

# Load dataset (WMT14 English-German)
dataset = load_dataset("wmt14", "de-en", split="train[:1%]")  # Subset for testing

# Preprocess dataset
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["de"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Split into train and eval datasets
dataset_split = dataset.train_test_split(test_size=0.1)  # Use 10% for evaluation
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"].select(range(500))

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    #gradient_accumulation_steps=4,
    save_total_limit=2,
    num_train_epochs=5,  # Increased number of epochs
    predict_with_generate=False,  # Speeds up evaluation
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=10,
)

# Metric for evaluation
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU expects a list of references for each prediction
    decoded_labels = [[label] for label in decoded_labels]

    # Calculate BLEU
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    # Ensure scalar-only logging
    return {"bleu": float(bleu_score["score"])}

# Data collator
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

# Track memory usage and training time
if torch.cuda.is_available():
    memory_start = torch.cuda.memory_allocated()
    print(f"Initial GPU Memory Usage: {memory_start / 1024**2:.2f} MB")
else:
    print("No GPU detected. Running on CPU.")

start_time = time.time()

# Training with epoch-wise BLEU logging
print("Starting training...")
for epoch in range(training_args.num_train_epochs):
    print(f"\nEpoch {epoch + 1}/{training_args.num_train_epochs}")
    trainer.train()
    print(f"Completed epoch {epoch + 1}. Evaluating...")
    eval_results = trainer.evaluate()
    print(f"BLEU Score After Epoch {epoch + 1}: {eval_results.get('bleu', 'N/A')}")

end_time = time.time()

if torch.cuda.is_available():
    memory_end = torch.cuda.memory_allocated()
    print(f"Final GPU Memory Usage: {memory_end / 1024**2:.2f} MB")
else:
    memory_end = 0

training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Evaluate the model on test set
print("Final evaluation...")
final_results = trainer.evaluate()
final_bleu_score = final_results.get("bleu", "N/A")
print(f"Final BLEU Score: {final_bleu_score}")

# Log results
print("\nSummary:")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Initial GPU Memory Usage: {memory_start / 1024**2:.2f} MB")
print(f"Final GPU Memory Usage: {memory_end / 1024**2:.2f} MB")
print(f"Final BLEU Score: {final_bleu_score}")
