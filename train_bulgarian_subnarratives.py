from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
from scipy.special import expit
from datasets import Dataset
import json

# --- Prepare Subnarrative Labels ---
def prepare_labels_for_subnarratives(training_data, all_labels):
    subnarratives_only = [label for label in all_labels if label["type"] == "S"]
    label_to_idx = {label["label"]: idx for idx, label in enumerate(subnarratives_only)}

    num_classes = len(label_to_idx)
    binary_labels = np.zeros((len(training_data), num_classes))

    for i, article in enumerate(training_data):
        subnarratives = article["subnarratives"]
        indices = [label_to_idx[label] for label in subnarratives if label in label_to_idx]
        binary_labels[i, indices] = 1

    texts = [article["content"] for article in training_data]
    return texts, binary_labels, label_to_idx

# --- Tokenization ---
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

# --- Metrics ---
def compute_metrics(pred):
    logits, labels = pred
    probabilities = expit(logits)
    predictions = (probabilities > 0.5).astype(int)
    f1 = f1_score(labels, predictions, average="macro", zero_division=1)
    return {"f1_macro": f1}

# --- Training with Repeated KFold ---
# --- Training with Repeated KFold ---
def train_with_repeated_kfold_and_save(texts, labels):
    dataset = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    dataset = dataset.map(tokenize, batched=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)  # Adjusted to 10 repeats for consistency
    labels_flat = labels.argmax(axis=1)

    all_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(labels)), labels_flat)):
        print(f"\n=== Fold {fold+1} ===")
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base", num_labels=labels.shape[1]
        )

        # Define output directories
        output_dir = f"/content/drive/MyDrive/bulgarian_subnarrative_model/results_fold_{fold}"
        logging_dir = f"/content/drive/MyDrive/bulgarian_subnarrative_model/logs_fold_{fold}"

        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,  # Save results to Google Drive
            logging_dir=logging_dir,  # Save logs to Google Drive
            per_device_train_batch_size=8,  # Increased batch size for efficiency
            per_device_eval_batch_size=8,
            num_train_epochs=100,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=100,
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            learning_rate=5e-5,  # Adjusted learning rate for stability
            lr_scheduler_type="linear ",
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )

        trainer.train()

        # Compute F1 on validation set
        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions
        probabilities = expit(logits)
        predicted_labels = (probabilities > 0.5).astype(int)

        f1 = f1_score(val_dataset["label"], predicted_labels, average="macro", zero_division=1)
        all_f1_scores.append(f1)
        print(f"F1 Score for fold {fold+1}: {f1}")

        # Save the fold model and tokenizer
        model.save_pretrained(f"{output_dir}/model_checkpoint_f1_{f1:.4f}")
        tokenizer.save_pretrained(f"{output_dir}/tokenizer_checkpoint_f1_{f1:.4f}")

    mean_f1 = np.mean(all_f1_scores)
    print(f"\n=== Mean F1 Score (RepeatedStratifiedKFold): {mean_f1} ===")


    # Save the final best model
    final_output_dir = "/content/drive/MyDrive/bulgarian_subnarrative_model"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Subnarrative model and tokenizer saved to {final_output_dir}.")

    # Save final metrics
    with open("/content/drive/MyDrive/bulgarian_subnarrative_model/final_metrics.json", "w") as f:
        json.dump({"mean_f1": mean_f1, "fold_f1_scores": all_f1_scores}, f, indent=4)
    return mean_f1


# --- Main Script ---
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data", "bulgarian_training_dataset.json")
    labels_path = os.path.join(current_dir, "data", "bulgarian_all_labels.json")

    print("Loading data...")
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)["labels"]

    print("Preparing subnarrative labels...")
    texts, labels, label_to_idx = prepare_labels_for_subnarratives(training_data, all_labels)

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    print("Training model...")
    mean_f1 = train_with_repeated_kfold_and_save(texts, labels)
    print(f"Final Mean F1 Score: {mean_f1}")
