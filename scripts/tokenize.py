# scripts/04_tokenize.py
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer
import yaml

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Tokenization function ----------
def tokenize_and_save(
    df: pd.DataFrame,
    tokenizer_name: str,
    max_length: int,
    output_path: Path,
    token_mode: str,
    project_id_map: dict
) -> None:
    """
    Tokenize source code and save tokenized tensors with labels.

    Args:
        df: DataFrame with columns ['project', 'source', 'label']
        tokenizer_name: Huggingface tokenizer identifier
        max_length: maximum token length (with truncation/padding)
        output_path: path to save the torch file
        token_mode: 'none', 'name', or 'id' for project tokens
        project_id_map: mapping from project name to numeric ID
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Loaded tokenizer '{tokenizer_name}'")

    texts = []
    labels = []
    for _, row in df.iterrows():
        source = row['source']
        if token_mode == 'none':
            text = source
        else:
            if token_mode == 'name':
                token = f"[PROJECT_{row['project'].upper()}] "
            else:  # 'id'
                pid = project_id_map.get(row['project'], 0)
                token = f"[PROJECT_{pid}] "
            text = token + source
        texts.append(text)
        labels.append(int(row['label']))

    # Tokenize all texts
    logger.info(f"Tokenizing {len(texts)} samples (mode={token_mode}, max_length={max_length})...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels_tensor
        },
        output_path
    )
    logger.info(f"Tokenized dataset saved to {output_path}")

# ---------- Main entry point ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 04: Tokenize dataset with optional project tokens'
    )
    parser.add_argument(
        '--config', type=Path, default=Path('config.yaml'),
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        required=True,
        help='Path to the deduped & sampled dataset CSV (.csv)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='deepseek-ai/deepseek-coder-1.3b-instruct',
        help='Huggingface tokenizer identifier'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum sequence length for tokenization'
    )
    parser.add_argument(
        '--token-mode',
        type=str,
        choices=['none', 'name', 'id'],
        default='name',
        help='Project token mode: none (no token), name (token by project name), id (numeric ID token)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/tokenized/tokenized_dataset.pt'),
        help='Output path for tokenized torch file'
    )
    args = parser.parse_args()

    # Load project ID mapping from config
    cfg = yaml.safe_load(args.config.read_text())
    project_id_map = {
        name: idx + 1 for idx, name in enumerate(cfg.get('projects', {}).keys())
    }

    # Load dataset
    logger.info(f"Loading dataset from {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # Validate required columns
    for col in ['project', 'source', 'label']:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in dataset")
            exit(1)

    tokenize_and_save(
        df,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        output_path=args.output,
        token_mode=args.token_mode,
        project_id_map=project_id_map
    )