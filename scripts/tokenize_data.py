from datasets import load_from_disk
from transformers import AutoTokenizer

# Load processed dataset
dataset = load_from_disk("data/processed_financial_ner")

# Load tokenizer
model_name = "xlm-roberta-large-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Extract unique labels and create a mapping
all_labels = sorted(set(tag for tags in dataset["ner_tags"] for tag in tags))  # Flatten labels
label_map = {label: i for i, label in enumerate(all_labels)}

# Debug: Print label mapping
print("Label Map:", label_map)

# Function to tokenize and align labels
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    word_ids = tokenized_inputs.word_ids()  # Map tokens to original word indices
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # Ignore special tokens
        elif word_idx != previous_word_idx:
            labels.append(label_map.get(example["ner_tags"][word_idx], 0))  # Assign correct label ID
        else:
            labels.append(label_map.get(example["ner_tags"][word_idx], 0))  # Assign same label for subwords

        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

# Save processed dataset
tokenized_dataset.save_to_disk("data/tokenized_financial_ner")
print("✅ Tokenization complete. Saved tokenized dataset!")


# from datasets import load_from_disk

# dataset = load_from_disk("data/tokenized_financial_ner")
# print(dataset[0])