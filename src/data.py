# src/data.py

from datasets import load_dataset

def load_and_preprocess_data(dataset_name="glue", subset="mrpc", tokenizer=None, max_length=128):
    """
    Loads a dataset and tokenizes the texts.
    """
    dataset = load_dataset(dataset_name, subset)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                         truncation=True, padding="max_length", max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
