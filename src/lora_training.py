"""LoRA fine-tuning of DistilBERT for cancer/non-cancer classification."""

import os
import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, 'models', 'lora_model')
HISTORY_FILE = os.path.join(BASE_DIR, 'results', 'lora_history.json')
RESULTS_FILE = os.path.join(BASE_DIR, 'results', 'lora_test_results.json')
MODEL_NAME  = 'distilbert-base-uncased'


# ---------------------------------------------------------------------------
# Custom callback: captures per-epoch train loss cleanly
# ---------------------------------------------------------------------------
class EpochLossCallback(TrainerCallback):
    def __init__(self):
        self._step_losses: list[float] = []
        self.epoch_train_losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and 'eval_loss' not in logs:
            self._step_losses.append(logs['loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._step_losses:
            self.epoch_train_losses.append(float(np.mean(self._step_losses)))
        else:
            self.epoch_train_losses.append(0.0)
        self._step_losses = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tokenize_df(df, tokenizer, max_length: int = 512):
    dataset = Dataset.from_pandas(
        df[['cleaned_text', 'label']].rename(columns={'cleaned_text': 'text'})
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized


def _compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy':  float(acc),
        'f1':        float(f1),
        'precision': float(precision),
        'recall':    float(recall),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lora(
    train_df,
    val_df,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: 'non_cancer', 1: 'cancer'},
        label2id={'non_cancer': 0, 'cancer': 1},
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=['q_lin'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_train = _tokenize_df(train_df, tokenizer)
    tokenized_val   = _tokenize_df(val_df,   tokenizer)

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_dir=os.path.join(BASE_DIR, 'logs', 'lora'),
        logging_steps=10,
        report_to='none',
        fp16=use_fp16,
    )

    loss_callback = EpochLossCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[loss_callback],
    )

    print("\n=== Starting LoRA Fine-Tuning ===")
    trainer.train()

    # Build history from log records
    eval_logs = [l for l in trainer.state.log_history if 'eval_loss' in l]
    history = {
        'epochs':        [l['epoch'] for l in eval_logs],
        'train_loss':    loss_callback.epoch_train_losses[:len(eval_logs)],
        'val_loss':      [l['eval_loss'] for l in eval_logs],
        'val_accuracy':  [l.get('eval_accuracy', 0.0) for l in eval_logs],
        'val_f1':        [l.get('eval_f1', 0.0) for l in eval_logs],
        'val_precision': [l.get('eval_precision', 0.0) for l in eval_logs],
        'val_recall':    [l.get('eval_recall', 0.0) for l in eval_logs],
    }

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved → {HISTORY_FILE}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved → {OUTPUT_DIR}")

    return model, tokenizer, trainer, history


# ---------------------------------------------------------------------------
# Evaluation on held-out test set
# ---------------------------------------------------------------------------
def evaluate_on_test(model, tokenizer, test_df):
    tokenized_test = _tokenize_df(test_df, tokenizer)

    eval_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=8,
        report_to='none',
    )

    evaluator = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=_compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    raw_preds = evaluator.predict(tokenized_test)
    metrics   = _compute_metrics(raw_preds)
    probs     = torch.softmax(torch.tensor(raw_preds.predictions), dim=-1).numpy()
    preds     = raw_preds.predictions.argmax(-1)

    test_results = {
        'metrics':       metrics,
        'predictions':   preds.tolist(),
        'probabilities': probs.tolist(),
        'labels':        test_df['label'].tolist(),
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nLoRA test results → {RESULTS_FILE}")
    print(f"Metrics: {metrics}")
    return test_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from data_loader import load_raw_data, preprocess, split_data, save_splits, load_splits, SPLITS_DIR

    if not os.path.exists(os.path.join(SPLITS_DIR, 'train.csv')):
        print("Creating data splits...")
        df = load_raw_data()
        df = preprocess(df)
        train_df, val_df, test_df = split_data(df)
        save_splits(train_df, val_df, test_df)

    train_df, val_df, test_df = load_splits()
    model, tokenizer, trainer, history = train_lora(train_df, val_df)
    evaluate_on_test(model, tokenizer, test_df)
    print("\nLoRA training complete!")
