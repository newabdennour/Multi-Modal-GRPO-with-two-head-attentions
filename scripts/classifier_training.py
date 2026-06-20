import os
import argparse
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train prompt classifier for modality routing")
    parser.add_argument("--data-path", default="data/classifier_dataset.csv",
                        help="Path to the training CSV")
    parser.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3",
                        help="HuggingFace model name")
    parser.add_argument("--save-path", default="./classifier_output",
                        help="Directory to save trained model")
    parser.add_argument("--num-classes", type=int, default=11,
                        help="Number of output classes (must match N_MODALITIES)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max tokenization length")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df = df[["prompt", "label"]].dropna()
    print(f"Loaded {len(df)} examples from {args.data_path}")

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    label_map = {label: int(idx) for label, idx in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    print(f"Labels: {label_map}")

    train_df, val_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label_id"], random_state=args.seed
    )
    print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label_id", "labels")
    val_dataset = val_dataset.rename_column("label_id", "labels")

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_path, "trainer_output"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="no",
        logging_steps=50,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal metrics: {metrics}")

    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"\nClassifier saved to: {args.save_path}")


if __name__ == "__main__":
    main()
