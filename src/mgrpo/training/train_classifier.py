import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from mgrpo.config import CLASSIFIER_DATASET_PATH, CLASSIFIER_PATH, MODALITY_MAP

# Invert MODALITY_MAP to map from string to integer ID
REVERSE_MODALITY_MAP = {v.lower(): k for k, v in MODALITY_MAP.items()}

def train_classifier():
    if not CLASSIFIER_DATASET_PATH.exists():
        print(f"Error: {CLASSIFIER_DATASET_PATH} not found.")
        return

    print(f"Loading dataset from {CLASSIFIER_DATASET_PATH}...")
    df = pd.read_csv(CLASSIFIER_DATASET_PATH)

    # Validate columns
    if 'prompt' not in df.columns or 'label' not in df.columns:
        print("Error: Dataset must contain 'prompt' and 'label' columns.")
        return

    # Map string labels to modality IDs
    df['label_id'] = df['label'].str.lower().map(REVERSE_MODALITY_MAP)
    
    unmapped = df[df['label_id'].isna()]
    if not unmapped.empty:
        print(f"Warning: Found {len(unmapped)} rows with unknown labels. Dropping them.")
        df = df.dropna(subset=['label_id'])

    df['label_id'] = df['label_id'].astype(int)

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # We use a lightweight model by default, e.g., distilbert
    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    num_labels = max(MODALITY_MAP.keys()) + 1
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={k: v for k, v in MODALITY_MAP.items()},
        label2id=REVERSE_MODALITY_MAP
    )

    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Ensure the label column is correctly named for Transformers
    train_dataset = train_dataset.rename_column("label_id", "labels")
    val_dataset = val_dataset.rename_column("label_id", "labels")
    
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    out_dir = CLASSIFIER_PATH if not CLASSIFIER_PATH.startswith("/") else "./outputs/classifier"
    print(f"Training classifier. Output will be saved to {out_dir}")

    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print(f"Training complete. Saving final model to {out_dir}...")
    trainer.save_model(out_dir)
