from datasets import load_dataset, Dataset
import json
import re

# Load dataset
dataset = load_dataset("Josephgflowers/Financial-NER-NLP")

def extract_ner_data(example):
    try:
        user_text = example.get("user", "").strip()  # Extract financial text
        assistant_response = example.get("assistant", "").strip()  # Extract entity annotations

        # Skip invalid examples
        if not user_text or not assistant_response or "No XBRL associated data." in assistant_response:
            return {"tokens": [], "ner_tags": []}  # Empty case but still keeps valid structure

        # Convert assistant response from string to JSON safely
        try:
            labels = json.loads(assistant_response.replace("'", "\""))
        except json.JSONDecodeError:
            print(f"Skipping due to JSON decode error: {assistant_response}")
            return {"tokens": [], "ner_tags": []}

        # Tokenize text into words while keeping numbers and currency symbols
        tokens = re.findall(r"\b\w+\b|\$?\d+[.,]?\d*\b", user_text)
        ner_tags = ["O"] * len(tokens)  # Default all tokens as 'O' (outside entity)

        entity_found = False  # Track if we find at least one entity

        # Assign entity labels
        for entity, values in labels.items():
            for value in values:
                # Fuzzy match entity value in text
                value_tokens = re.findall(r"\b\w+\b|\$?\d+[.,]?\d*\b", value)
                value_length = len(value_tokens)

                for i in range(len(tokens) - value_length + 1):
                    if tokens[i:i+value_length] == value_tokens:
                        ner_tags[i] = f"B-{entity}"  # Beginning of entity
                        for j in range(1, value_length):
                            ner_tags[i+j] = f"I-{entity}"  # Inside entity
                        entity_found = True  # We found at least one entity

        # Debugging: Show only if we found an entity
        if entity_found:
            print(f"\n✅ Found Entity in Example:\nTokens: {tokens[:10]}\nNER Tags: {ner_tags[:10]}\n")
        
        return {"tokens": tokens, "ner_tags": ner_tags}

    except Exception as e:
        print(f"Error processing example: {str(e)}")
        return {"tokens": [], "ner_tags": []}  # Keep valid format even on error

# Apply transformation
processed_data = dataset["train"].map(extract_ner_data)

# **Ensure the dataset contains at least one valid example**
valid_data = [x for x in processed_data if x["tokens"]]

if not valid_data:
    print("❌ No valid NER data found. Consider switching datasets.")
else:
    # Convert to Dataset format and save
    processed_dataset = Dataset.from_dict({
        "tokens": [x["tokens"] for x in valid_data],
        "ner_tags": [x["ner_tags"] for x in valid_data]
    })
    processed_dataset.save_to_disk("data/processed_financial_ner")
    print("\n✅ Preprocessing complete. Saved processed dataset!")



# from datasets import load_from_disk

# # Load processed dataset
# dataset = load_from_disk("data/processed_financial_ner")

# # Print sample
# print(dataset[1])
