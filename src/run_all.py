"""
Orchestration script — runs the full pipeline:
  1. Data loading & splitting (70 / 15 / 15)
  2. LoRA fine-tuning
  3. Knowledge distillation (GPT-4o teacher → DistilBERT student)
  4. Side-by-side comparison printed to console
"""

import os
import json
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Step 1: Data
# ---------------------------------------------------------------------------
def step_data():
    _banner("STEP 1: Data Loading & Splitting")
    from data_loader import (
        load_raw_data, preprocess, split_data, save_splits, load_splits, SPLITS_DIR
    )
    train_path = os.path.join(SPLITS_DIR, 'train.csv')
    if os.path.exists(train_path):
        print("Data splits already exist — loading cached splits.")
        return load_splits()

    df = load_raw_data()
    df = preprocess(df)
    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Step 2: LoRA
# ---------------------------------------------------------------------------
def step_lora(train_df, val_df, test_df):
    _banner("STEP 2: LoRA Fine-Tuning (DistilBERT)")
    from lora_training import train_lora, evaluate_on_test, RESULTS_FILE

    if os.path.exists(RESULTS_FILE):
        print("LoRA results already exist — skipping training.")
        with open(RESULTS_FILE) as f:
            return json.load(f)

    model, tokenizer, trainer, history = train_lora(train_df, val_df)
    return evaluate_on_test(model, tokenizer, test_df)


# ---------------------------------------------------------------------------
# Step 3: Distillation
# ---------------------------------------------------------------------------
def step_distillation(train_df, val_df, test_df):
    _banner("STEP 3: Knowledge Distillation (GPT-4o → DistilBERT)")
    from distillation_training import train_distillation, evaluate_on_test, RESULTS_FILE

    if os.path.exists(RESULTS_FILE):
        print("Distillation results already exist — skipping training.")
        with open(RESULTS_FILE) as f:
            return json.load(f)

    model, tokenizer, history = train_distillation(train_df, val_df)
    return evaluate_on_test(model, tokenizer, test_df)


# ---------------------------------------------------------------------------
# Step 4: Comparison
# ---------------------------------------------------------------------------
def print_comparison(lora_results: dict, dist_results: dict):
    _banner("RESULTS COMPARISON")

    lora_m = lora_results['metrics']
    dist_m = dist_results['metrics']

    # normalise key names (Trainer prefixes with 'eval_')
    def _get(m, key):
        return m.get(key, m.get(f'eval_{key}', 0.0))

    rows = ['accuracy', 'f1', 'precision', 'recall']
    header = f"{'Metric':<15} {'LoRA':>10} {'Distillation':>14} {'Δ (Dist−LoRA)':>15}"
    print(header)
    print("-" * len(header))
    for row in rows:
        lv = _get(lora_m, row)
        dv = _get(dist_m, row)
        delta = dv - lv
        sign  = '+' if delta >= 0 else ''
        print(f"{row:<15} {lv:>10.4f} {dv:>14.4f} {sign}{delta:>14.4f}")

    results_summary = {
        'lora':        {r: _get(lora_m, r) for r in rows},
        'distillation':{r: _get(dist_m, r) for r in rows},
    }
    summary_path = os.path.join(BASE_DIR, 'results', 'comparison_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    train_df, val_df, test_df = step_data()
    lora_results = step_lora(train_df, val_df, test_df)
    dist_results = step_distillation(train_df, val_df, test_df)
    print_comparison(lora_results, dist_results)

    _banner("ALL STEPS COMPLETE")
    print("Open notebooks/results.ipynb to view plots and detailed analysis.")
