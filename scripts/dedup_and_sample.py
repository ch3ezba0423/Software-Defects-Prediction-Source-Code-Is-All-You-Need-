import argparse
import logging
import hashlib
import random
from pathlib import Path
import pandas as pd
import yaml

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Utility functions ----------
def load_config(path: Path) -> dict:
    """Load the YAML configuration containing project paths and metrics."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_hash(text: str) -> str:
    """Compute an MD5 hash for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ---------- Deduplication and Sampling ----------
def build_dataset(cfg: dict, processed_root: Path) -> pd.DataFrame:
    """Build a unified DataFrame with columns: project, version, fqn, label, source, hash."""
    records = []
    for project, paths in cfg['projects'].items():
        for src_path, metrics_path in zip(paths['sources'], paths['metrics']):
            version = Path(src_path).name
            proc_dir = processed_root / project / version
            metrics_df = pd.read_csv(metrics_path, index_col=0)
            # Determine bug column
            if 'bug' in metrics_df.columns:
                bug_col = 'bug'
            else:
                bug_col = metrics_df.columns[-1]

            if not proc_dir.exists():
                logger.warning(f"Processed directory not found: {proc_dir}")
                continue
            for file in proc_dir.rglob('*.java'):
                rel = file.relative_to(proc_dir)
                fqn = rel.with_suffix('').as_posix().replace('/', '.')
                try:
                    source = file.read_text(encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Failed reading {file}: {e}")
                    continue
                # metric lookup
                if fqn not in metrics_df.index:
                    continue
                bug_count = metrics_df.at[fqn, bug_col]
                label = 1 if bug_count > 0 else 0
                source_hash = compute_hash(source)
                records.append({
                    'project': project,
                    'version': version,
                    'fqn': fqn,
                    'label': label,
                    'source': source,
                    'hash': source_hash
                })
    df = pd.DataFrame(records)
    return df


def dedup_and_balance(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Remove duplicate sources and balance classes via undersampling."""
    # Deduplicate by hash
    before = len(df)
    df_unique = df.drop_duplicates(subset=['hash'])
    logger.info(f"Deduplicated: {before} -> {len(df_unique)} records (unique sources)")

    # Balance labels
    bugs = df_unique[df_unique['label'] == 1]
    nonbugs = df_unique[df_unique['label'] == 0]
    n_bugs = len(bugs)
    n_nonbugs = len(nonbugs)
    if n_nonbugs > n_bugs:
        nonbugs_sampled = nonbugs.sample(n_bugs, random_state=seed)
        df_balanced = pd.concat([bugs, nonbugs_sampled], ignore_index=True)
        logger.info(f"Balanced dataset: {len(bugs)} bugs and {len(nonbugs_sampled)} non-bugs")
    else:
        df_balanced = df_unique
        logger.info("Dataset already balanced or has fewer non-bugs than bugs; no undersampling applied")

    # Shuffle
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_balanced

# ---------- Main entry point ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 03: Deduplicate and sample processed Java code dataset'
    )
    parser.add_argument(
        '--config', type=Path, default=Path('config.yaml'),
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--processed-root', type=Path, default=Path('data/processed'),
        help='Root directory for preprocessed Java files'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('data/processed/unified_dataset.xlsx'),
        help='Output path for the unified deduped and sampled dataset (.xlsx)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sampling'
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = build_dataset(cfg, args.processed_root)
    df_balanced = dedup_and_balance(df, seed=args.seed)
    # Save to Excel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_excel(args.output, index=False)
    logger.info(f"Unified deduped & sampled dataset saved to {args.output}")
    # Also save CSV
    csv_out = args.output.with_suffix('.csv')
    df_balanced.to_csv(csv_out, index=False)
    logger.info(f"Also saved CSV version to {csv_out}")
