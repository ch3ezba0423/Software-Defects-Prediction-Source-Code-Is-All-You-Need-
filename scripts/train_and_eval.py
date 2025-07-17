import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AdamW,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split

# Configure logger
torch.manual_seed(42)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(input_csv: Path, tokenized_path: Path):
    """Load dataset CSV and tokenized tensors, return project list and tensors."""
    df = pd.read_csv(input_csv)
    data = torch.load(tokenized_path)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']
    projects = df['project'].tolist()
    return projects, input_ids, attention_mask, labels


def train_and_evaluate(
    projects, input_ids, attention_mask, labels,
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_accum_steps: int,
    early_stop_patience: int,
    scheduler_warmup: float = 0.1,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unique_projects = sorted(set(projects))
    results = []

    for proj in unique_projects:
        logger.info(f"Starting LOPO fold: test project = {proj}")
        # Split indices for LOPOCV
        idx_test = [i for i, p in enumerate(projects) if p == proj]
        idx_train = [i for i, p in enumerate(projects) if p != proj]
        # Stratified train/val split
        train_idx, val_idx = train_test_split(
            idx_train,
            test_size=0.2,
            stratify=[labels[i].item() for i in idx_train],
            random_state=42
        )

        # Prepare DataLoaders
        def subset(indices):
            return TensorDataset(
                input_ids[indices],
                attention_mask[indices],
                labels[indices]
            )
        train_loader = DataLoader(subset(train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(subset(val_idx), batch_size=batch_size)
        test_loader = DataLoader(subset(idx_test), batch_size=batch_size)

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(device)

        optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        total_steps = epochs * len(train_loader) // grad_accum_steps
        warmup_steps = int(total_steps * scheduler_warmup)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        scaler = torch.cuda.amp.GradScaler()

        # Early stopping setup
        best_val_f1 = 0.0
        no_improve_count = 0

        # Training loop with early stopping
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                input_ids_b, attn_b, labels_b = [b.to(device) for b in batch]
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids_b,
                        attention_mask=attn_b,
                        labels=labels_b
                    )
                    loss = outputs.loss / grad_accum_steps
                scaler.scale(loss).backward()
                total_loss += loss.item()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch}/{epochs} - avg train loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids_b, attn_b, labels_b = [b.to(device) for b in batch]
                    logits = model(input_ids=input_ids_b, attention_mask=attn_b).logits
                    preds = torch.argmax(logits, dim=-1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels_b.cpu().numpy())
            prec, rec, f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average='binary'
            )
            try:
                auc = roc_auc_score(val_labels, val_preds)
            except ValueError:
                auc = float('nan')
            logger.info(
                f"Validation Epoch {epoch} - P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
            )

            # Check early stopping
            if f1 > best_val_f1:
                best_val_f1 = f1
                no_improve_count = 0
                # Save checkpoint of best model
                best_model_dir = output_dir / f"fold_{proj}" / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_model_dir)
            else:
                no_improve_count += 1
                logger.info(f"No improvement count: {no_improve_count}/{early_stop_patience}")
                if no_improve_count >= early_stop_patience:
                    logger.info("Early stopping triggered")
                    break

        # Test evaluation on last or best model
        # Load best model if available
        best_model_path = output_dir / f"fold_{proj}" / "best_model"
        if best_model_path.exists():
            model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(device)

        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids_b, attn_b, labels_b = [b.to(device) for b in batch]
                logits = model(input_ids=input_ids_b, attention_mask=attn_b).logits
                preds = torch.argmax(logits, dim=-1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels_b.cpu().numpy())
        cm = confusion_matrix(test_labels, test_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='binary'
        )
        try:
            auc = roc_auc_score(test_labels, test_preds)
        except ValueError:
            auc = float('nan')

        logger.info(
            f"Test project {proj} - CM: {cm.tolist()}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
        )
        results.append({
            'project': proj,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        })

    # Save results summary
    results_df = pd.DataFrame(results)
    summary_csv = output_dir / 'LOPOCV_results.csv'
    results_df.to_csv(summary_csv, index=False)
    logger.info(f"LOPOCV summary saved to {summary_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 05: LOPOCV training & evaluation with early stopping'
    )
    parser.add_argument(
        '--input-csv', type=Path, required=True,
        help='Path to deduped & sampled dataset CSV'
    )
    parser.add_argument(
        '--tokenized', type=Path, required=True,
        help='Path to tokenized .pt dataset file'
    )
    parser.add_argument(
        '--model-name', type=str,
        default='deepseek-ai/deepseek-coder-1.3b-instruct',
        help='Huggingface model identifier for sequence classification'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('outputs/train'),
        help='Directory to save models and metrics'
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=3e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--grad-accum-steps', type=int, default=16,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--early-stop-patience', type=int, default=3,
        help='Patience epochs for early stopping based on validation F1'
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    projects, input_ids, attention_mask, labels = load_data(
        args.input_csv, args.tokenized
    )
    train_and_evaluate(
        projects, input_ids, attention_mask, labels,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        early_stop_patience=args.early_stop_patience
    )