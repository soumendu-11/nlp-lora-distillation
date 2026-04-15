"""
Knowledge Distillation: GPT-4o (Azure) teacher → DistilBERT student.

Flow
----
1. Call Azure GPT-4o for each training sample → soft probability [P(non-cancer), P(cancer)]
2. Cache soft labels to disk (skips already-processed samples on re-runs)
3. Train DistilBERT student with:
       L = α · T² · KL(student_log_softmax/T ‖ teacher_probs)
         + (1−α) · CrossEntropy(logits, hard_label)
4. Evaluate on test set with standard metrics
"""

import os
import json
import time
import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

OUTPUT_DIR        = os.path.join(BASE_DIR, 'models', 'distillation_model')
SOFT_LABELS_CACHE = os.path.join(BASE_DIR, 'data_splits', 'soft_labels_cache.json')
HISTORY_FILE      = os.path.join(BASE_DIR, 'results', 'distillation_history.json')
RESULTS_FILE      = os.path.join(BASE_DIR, 'results', 'distillation_test_results.json')
MODEL_NAME        = 'distilbert-base-uncased'

TEMPERATURE = 4.0   # distillation temperature
ALPHA       = 0.7   # weight for KL loss; (1-alpha) for hard-label CE


# ---------------------------------------------------------------------------
# GPT-4o teacher: soft label generation
# ---------------------------------------------------------------------------
def _get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv('AZURE_API_KEY'),
        api_version=os.getenv('API_VERSION', '2025-01-01-preview'),
        azure_endpoint=os.getenv('AZURE_ENDPOINT'),
    )


def get_gpt4o_soft_labels(
    texts: List[str],
    cache_file: str = SOFT_LABELS_CACHE,
) -> List[List[float]]:
    """
    Return soft labels [[P(non-cancer), P(cancer)], ...] for every text.
    Results are cached by content hash so the API is only called once per sample.
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    cache: dict = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)

    deployment = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
    client     = _get_azure_client()

    soft_labels: List[List[float]] = []
    new_calls = 0

    for i, text in enumerate(tqdm(texts, desc='GPT-4o soft labels')):
        key = str(hash(text))
        if key in cache:
            soft_labels.append(cache[key])
            continue

        prompt = (
            "You are a biomedical text classifier. "
            "Given the abstract below, output the probability that it is cancer-related "
            "vs non-cancer-related. "
            "Respond ONLY with a valid JSON object, no markdown:\n"
            '{"cancer_probability": <float 0-1>, "non_cancer_probability": <float 0-1>}\n'
            "The two probabilities must sum to 1.0.\n\n"
            f"Abstract: {text[:2000]}"
        )

        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{'role': 'user', 'content': prompt}],
                response_format={'type': 'json_object'},
                temperature=0,
                max_tokens=60,
            )
            result = json.loads(response.choices[0].message.content)
            cp  = float(result.get('cancer_probability', 0.5))
            ncp = float(result.get('non_cancer_probability', 1.0 - cp))
            total = cp + ncp
            if total > 0:
                cp, ncp = cp / total, ncp / total
            else:
                cp, ncp = 0.5, 0.5
            label = [ncp, cp]          # index-0 = non-cancer, index-1 = cancer
        except Exception as e:
            print(f"  Warning: GPT-4o call failed for sample {i}: {e}")
            label = [0.5, 0.5]

        soft_labels.append(label)
        cache[key] = label
        new_calls += 1

        if new_calls % 20 == 0:
            with open(cache_file, 'w') as f:
                json.dump(cache, f)

        time.sleep(0.05)   # gentle rate-limit

    with open(cache_file, 'w') as f:
        json.dump(cache, f)

    print(f"Soft-label cache: {new_calls} new API calls | total cached: {len(cache)}")
    return soft_labels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class DistillDataset(TorchDataset):
    def __init__(self, encodings, hard_labels: List[int], soft_labels: List[List[float]]):
        self.input_ids      = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.hard_labels    = hard_labels
        self.soft_labels    = torch.tensor(soft_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.hard_labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'hard_label':     torch.tensor(self.hard_labels[idx], dtype=torch.long),
            'soft_label':     self.soft_labels[idx],
        }


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------
def distillation_loss(
    student_logits: torch.Tensor,
    soft_labels: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = TEMPERATURE,
    alpha: float = ALPHA,
) -> torch.Tensor:
    # KL divergence between student (temperature-scaled) and teacher distributions
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs     = soft_labels.clamp(min=1e-8)
    teacher_probs     = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)

    kl = torch.nn.functional.kl_div(
        student_log_probs, teacher_probs, reduction='batchmean'
    ) * (temperature ** 2)

    ce = torch.nn.functional.cross_entropy(student_logits, hard_labels)
    return alpha * kl + (1.0 - alpha) * ce


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _encode(texts: List[str], tokenizer) -> dict:
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )


def train_distillation(
    train_df,
    val_df,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

    print("\n=== Generating GPT-4o Soft Labels ===")
    train_soft = get_gpt4o_soft_labels(train_df['cleaned_text'].tolist())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = _encode(train_df['cleaned_text'].tolist(), tokenizer)
    val_enc   = _encode(val_df['cleaned_text'].tolist(),   tokenizer)

    # Validation uses uniform soft labels (only hard-CE evaluated during val)
    val_soft = [[0.5, 0.5]] * len(val_df)

    train_dataset = DistillDataset(train_enc, train_df['label'].tolist(), train_soft)
    val_dataset   = DistillDataset(val_enc,   val_df['label'].tolist(),   val_soft)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    device = _get_device()
    print(f"Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: 'non_cancer', 1: 'cancer'},
        label2id={'non_cancer': 0, 'cancer': 1},
    ).to(device)

    optimizer     = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps   = len(train_loader) * epochs
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    history = {
        'epochs':       [],
        'train_loss':   [],
        'val_loss':     [],
        'val_accuracy': [],
        'val_f1':       [],
    }

    best_f1       = 0.0
    patience      = 3
    patience_cnt  = 0

    print("\n=== Starting Distillation Training ===")
    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            optimizer.zero_grad()
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            hard   = batch['hard_label'].to(device)
            soft   = batch['soft_label'].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits
            loss   = distillation_loss(logits, soft, hard)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        # -- Validate --
        model.eval()
        total_val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                hard = batch['hard_label'].to(device)

                logits = model(input_ids=ids, attention_mask=mask).logits
                loss   = torch.nn.functional.cross_entropy(logits, hard)
                total_val_loss += loss.item()
                all_preds.extend(logits.argmax(-1).cpu().numpy())
                all_labels.extend(hard.cpu().numpy())

        avg_val  = total_val_loss / len(val_loader)
        val_acc  = float(accuracy_score(all_labels, all_preds))
        val_f1   = float(f1_score(all_labels, all_preds, average='binary'))

        history['epochs'].append(epoch)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ↑ Best model saved (F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved → {HISTORY_FILE}")

    return model, tokenizer, history


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(model, tokenizer, test_df):
    device = _get_device()
    model  = model.to(device)
    model.eval()

    test_enc  = _encode(test_df['cleaned_text'].tolist(), tokenizer)
    test_soft = [[0.5, 0.5]] * len(test_df)
    test_ds   = DistillDataset(test_enc, test_df['label'].tolist(), test_soft)
    test_ld   = DataLoader(test_ds, batch_size=8, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_ld:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['hard_label']

            logits = model(input_ids=ids, attention_mask=mask).logits
            probs  = torch.softmax(logits, dim=-1)

            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy().tolist())

    acc = float(accuracy_score(all_labels, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    test_results = {
        'metrics': {
            'eval_accuracy':  acc,
            'eval_f1':        float(f1),
            'eval_precision': float(precision),
            'eval_recall':    float(recall),
        },
        'predictions':   [int(p) for p in all_preds],
        'probabilities': all_probs,
        'labels':        [int(l) for l in all_labels],
    }

    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nDistillation test results → {RESULTS_FILE}")
    print(f"Metrics: {test_results['metrics']}")
    return test_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from data_loader import (
        load_raw_data, preprocess, split_data, save_splits, load_splits, SPLITS_DIR
    )

    if not os.path.exists(os.path.join(SPLITS_DIR, 'train.csv')):
        print("Creating data splits...")
        df = load_raw_data()
        df = preprocess(df)
        train_df, val_df, test_df = split_data(df)
        save_splits(train_df, val_df, test_df)

    train_df, val_df, test_df = load_splits()
    model, tokenizer, history = train_distillation(train_df, val_df)
    evaluate_on_test(model, tokenizer, test_df)
    print("\nDistillation training complete!")
