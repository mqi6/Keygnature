# src/trainer.py

import torch
from transformers import Trainer, TrainingArguments

def train_model(model, tokenized_dataset):
    """
    Train the model using Hugging Face's Trainer.
    """
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    
    trainer.train()
    return trainer
