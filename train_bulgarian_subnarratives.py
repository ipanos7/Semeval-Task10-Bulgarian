from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
from scipy.special import expit
from datasets import Dataset
import json

# --- Prepare Labels ---
def prepare_labels(training_data, all_labels):
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
def train_with_repeated_kfold_and_save_ensemble(texts, labels):
    dataset = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    dataset = dataset.map(tokenize, batched=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    labels_flat = labels.argmax(axis=1)

    all_predictions = []
    all_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(labels)), labels_flat)):
        print(f"\n=== Fold {fold+1} ===")
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base", num_labels=labels.shape[1]
        )

        training_args = TrainingArguments(
            output_dir=f"/content/drive/MyDrive/Bulgarian_Subnarratives/results_fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"/content/drive/MyDrive/Bulgarian_Subnarratives/logs_fold_{fold}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=100,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=3e-5,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            lr_scheduler_type="linear",
            logging_steps=100,
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

        # Retrieve the path of the best model for this fold
        best_model_path = trainer.state.best_model_checkpoint
        print(f"Best model for fold {fold+1} saved at: {best_model_path}")

        # Save the model to Google Drive for the current fold
        model.save_pretrained(f"/content/drive/MyDrive/Bulgarian_Subnarratives/models_fold_{fold}")
        tokenizer.save_pretrained(f"/content/drive/MyDrive/Bulgarian_Subnarratives/models_fold_{fold}")

        # Compute F1 on validation set
        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions
        probabilities = expit(logits)
        all_predictions.append(probabilities)

        predicted_labels = (probabilities > 0.5).astype(int)

        f1 = f1_score(val_dataset["label"], predicted_labels, average="macro", zero_division=1)
        all_f1_scores.append(f1)
        print(f"F1 Score for fold {fold+1}: {f1}")

    # Log all fold F1 scores
    print("\n=== Individual Fold F1 Scores ===")
    for fold, score in enumerate(all_f1_scores, 1):
        print(f"Fold {fold}: {score}")

    # Ensemble predictions by averaging
    ensemble_predictions = np.mean(all_predictions, axis=0)
    final_predictions = (ensemble_predictions > 0.5).astype(int)

    # Evaluate ensemble performance
    f1_ensemble = f1_score(labels, final_predictions, average="macro", zero_division=1)
    print(f"\n=== Ensemble F1 Score: {f1_ensemble} ===")

    best_fold = np.argmax(all_f1_scores)
    best_model_path = f"/content/drive/MyDrive/Bulgarian_Subnarratives/models_fold_{best_fold}"
    print(f"Best model across all folds is from fold {best_fold+1}, saved at: {best_model_path}")
    return best_model_path, f1_ensemble, all_f1_scores

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
    texts, labels, label_to_idx = prepare_labels(training_data, all_labels)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    print("Training with Repeated Stratified K-Fold and saving ensemble model...")
    f1_ensemble = train_with_repeated_kfold_and_save_ensemble(texts, labels)
    print(f"Final Ensemble F1 Score: {f1_ensemble}")
