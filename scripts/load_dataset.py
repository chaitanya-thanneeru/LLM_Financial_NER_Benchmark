# from datasets import load_dataset

# # Load Financial NER dataset
# dataset = load_dataset("roro/Financial-NER")

# # Print dataset details
# print(dataset)
# print(dataset["train"][0])  # Sample record

from datasets import load_dataset
from huggingface_hub import login

try:
    #log in with your token.
    dataset = load_dataset("Josephgflowers/Financial-NER-NLP")
    print(dataset)
except Exception as e:
    print(f"Error: {e}")

# Print dataset details
print(dataset)
print(dataset["train"][0])  # Sample record