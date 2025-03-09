from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "xlm-roberta-large-finetuned-conll03-english"  # Choose a strong baseline model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

print("✅ Model and tokenizer loaded successfully!")
