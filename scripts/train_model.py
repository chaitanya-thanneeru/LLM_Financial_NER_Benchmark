import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # (Optional) Try to disable MPS memory limit
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU & Force CPU

import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk, DatasetDict
from transformers import DataCollatorForTokenClassification, AutoTokenizer

# Load tokenized dataset
dataset = load_from_disk("data/tokenized_financial_ner")

# Split dataset if not already split
if not isinstance(dataset, DatasetDict):  # Check if the dataset has train/val/test splits
    dataset = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test
    dataset["validation"] = dataset["test"]  # Use test split as validation

# Get unique labels
unique_labels = set(label for example in dataset["train"]["labels"] for label in example)
num_labels = len(unique_labels)

# Load model with correct num_labels
model_name = "xlm-roberta-large-finetuned-conll03-english"
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,  # Use dataset-specific label count
    ignore_mismatched_sizes=True  # This allows reinitializing the classifier layer
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="models/financial_ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    gradient_accumulation_steps=4
)

# Define Trainer
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()
trainer.save_model("models/financial_ner")

print(f"✅ Fine-tuning complete. Model saved with {num_labels} labels!")


# pip install transformers[torch]
# pip install 'accelerate>=0.26.0' this is required