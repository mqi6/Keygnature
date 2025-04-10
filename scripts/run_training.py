# scripts/run_training.py

from src.model import load_model_and_tokenizer
from src.data import load_and_preprocess_data
from src.trainer import train_model

def main():
    model_name = "bert-base-uncased"
    num_labels = 2  # Change as needed for your task
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels)
    
    tokenized_dataset = load_and_preprocess_data(
        dataset_name="glue", subset="mrpc", tokenizer=tokenizer
    )
    
    trainer = train_model(model, tokenized_dataset)
    
    # Optionally save the model
    trainer.save_model("./final_model")

if __name__ == "__main__":
    main()
