import argparse
import logging
import re
import subprocess
from pathlib import Path
import yaml

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Regex patterns for comments and string literals
double_quote_str = re.compile(r'"(\\.|[^"\\])*"')
line_comment = re.compile(r'//.*')
block_comment = re.compile(r'/\*(?:.|\n)*?\*/')

# ---------- Utility functions ----------
def load_config(path: Path) -> dict:
    """Load the YAML configuration for project paths."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def remove_comments_and_literals(code: str) -> str:
    """Remove comments and replace string literals with a generic token."""
    # Remove block comments first
    code = block_comment.sub('', code)
    # Remove line comments
    code = line_comment.sub('', code)
    # Replace string literals with STRING token
    code = double_quote_str.sub('"STRING"', code)
    return code


def format_with_google_java(src_path: Path) -> None:
    """Run google-java-format in-place on the given file."""
    try:
        subprocess.run([
            'google-java-format', '-i', str(src_path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Formatting failed for {src_path}: {e}")

# ---------- Preprocessing step ----------
def process_project(project_name: str, cfg: dict, out_root: Path) -> None:
    """Clean and format Java files, then save to processed folder."""
    for src_path in cfg['sources']:
        src_root = Path(src_path)
        version = src_root.name
        dest_root = out_root / project_name / version
        for java_file in src_root.rglob('*.java'):
            rel_path = java_file.relative_to(src_root)
            dest_file = dest_root / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                code = java_file.read_text(encoding='utf-8')
                # Remove comments and literals
                cleaned = remove_comments_and_literals(code)
                # Write intermediate file
                dest_file.write_text(cleaned, encoding='utf-8')
                # Format in-place
                format_with_google_java(dest_file)
            except Exception as e:
                logger.warning(f"Failed to preprocess {java_file}: {e}")
        logger.info(f"Preprocessed project {project_name} version {version}")

# ---------- Main entry point ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 02: Preprocess Java files (remove comments and string literals, then format)'
    )
    parser.add_argument(
        '--config', type=Path, default=Path('config.yaml'),
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--out', type=Path, default=Path('data/processed'),
        help='Root directory for processed output'
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    for project, paths in cfg['projects'].items():
        process_project(project, paths, args.out)
    logger.info('Preprocessing complete for all projects.')