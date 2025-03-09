from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load fine-tuned model
model_path = "models/financial_ner"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Load pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Test on a sample financial sentence
text = "JP Morgan announced $10 billion investment in AI research."
results = ner_pipeline(text)

for entity in results:
    print(entity)
