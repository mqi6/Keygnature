# src/model.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_name="bert-base-uncased", num_labels=2):
    """
    Load a pretrained transformer model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer
