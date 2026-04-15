"""Data loading and preprocessing for cancer/non-cancer classification."""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
SPLITS_DIR = os.path.join(BASE_DIR, 'data_splits')


def load_raw_data(dataset_path: str = DATASET_PATH) -> pd.DataFrame:
    """Load cancer/non-cancer text files into a DataFrame."""
    data = []
    for label, folder in enumerate(['Non-Cancer', 'Cancer']):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                data.append({'text': text, 'label': label, 'filename': filename})

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples | Cancer: {(df['label']==1).sum()} | Non-Cancer: {(df['label']==0).sum()}")
    return df


def clean_text(text: str) -> str:
    """Remove XML-like tags, special chars, and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:()'\"!?% \n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df


def split_data(df: pd.DataFrame, random_state: int = 42):
    """
    Split data into train/val/test.
    Note: requested 75/15/15 sums to 105%, so we use 70/15/15 (adds to 100%).
    With 1000 samples: 700 train, 150 val, 150 test.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state, stratify=temp_df['label']
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir: str = SPLITS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'),   index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Splits saved to {output_dir}")


def load_splits(splits_dir: str = SPLITS_DIR):
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val_df   = pd.read_csv(os.path.join(splits_dir, 'val.csv'))
    test_df  = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
    print(f"Loaded splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


if __name__ == '__main__':
    df = load_raw_data()
    df = preprocess(df)
    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df)
