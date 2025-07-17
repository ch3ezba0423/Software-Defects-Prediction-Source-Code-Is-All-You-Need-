# uses config.yaml to extract full Java code
import argparse
import logging
from pathlib import Path
import pandas as pd
import javalang
import yaml

# Configure the logger
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


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load the metrics CSV and initialize the 'source' column."""
    df = pd.read_csv(csv_path, index_col=0)
    df['source'] = None
    return df


def find_java_files(root: Path) -> list:
    """Return a list of all .java files under the given root directory."""
    return list(root.rglob('*.java'))


def save_results(df: pd.DataFrame, out_path: Path) -> None:
    """Filter out entries without source and write results to an Excel file."""
    df_clean = df.dropna(subset=['source'])
    excel_path = out_path.with_suffix(out_path.suffix + '.xlsx')
    df_clean.to_excel(excel_path, index=False)
    logger.info(f"Results saved: {excel_path}")

# ---------- Code extraction ----------
def extract_classes_from_code(code: str, codelines: list) -> dict:
    """Parse Java code and return a dict mapping FQN to full source (with header)."""
    classes = {}
    try:
        tree = javalang.parse.parse(code)
        for path, class_node in tree.filter(javalang.tree.ClassDeclaration):
            package_name = path[0].package.name if path[0].package else ''
            for type_decl in path[0].types:
                type_decl.documentation = None
                fqn = f"{package_name}.{type_decl.name}" if package_name else type_decl.name
                classes[fqn] = build_source(type_decl, tree, codelines)
                # Handle inner classes
                for inner in type_decl.body:
                    if isinstance(inner, javalang.tree.ClassDeclaration):
                        inner_fqn = f"{fqn}.{inner.name}"
                        classes[inner_fqn] = build_source(inner, tree, codelines)
    except javalang.parser.JavaSyntaxError as e:
        logger.warning(f"SyntaxError during parsing: {e}")
    return classes


def build_source(class_node, tree, codelines) -> str:
    """Reconstruct the full source: header (package/imports) + tokenized class body."""
    startpos, endpos, startline, endline = get_class_boundaries(class_node, tree)
    snippet, _, _, _ = slice_source(startpos, endpos, startline, endline, codelines)
    tokens = list(javalang.tokenizer.tokenize(snippet))
    source_text = ''
    separator = ' '
    for tok in tokens:
        if isinstance(tok, (javalang.tokenizer.Separator, javalang.tokenizer.Operator)):
            separator = ''
        source_text += separator + tok.value
        separator = ' ' if isinstance(tok, (javalang.tokenizer.Identifier, javalang.tokenizer.Keyword)) else ''
    # Prepend header: lines before the class definition (package + imports)
    header = ''.join(codelines[: startline - 1]) if startline else ''
    return header + source_text


def get_class_boundaries(class_node, tree) -> tuple:
    """Determine the start and end positions for the class in the code.`"""
    startpos = endpos = None
    startline = endline = None
    for path, node in tree:
        if startpos is not None and class_node not in path:
            endpos = node.position
            endline = node.position.line if node.position else None
            break
        if startpos is None and node == class_node:
            startpos = node.position
            startline = node.position.line if node.position else None
    return startpos, endpos, startline, endline


def slice_source(startpos, endpos, startline, endline, codelines) -> tuple:
    """Extract source lines between startline and endline, balancing braces."""
    if not startpos:
        return '', None, None, None
    s_idx = startline - 1
    e_idx = endline - 1 if endpos else None
    snippet = '<ST>'.join(codelines[s_idx:e_idx])
    snippet = snippet[:snippet.rfind('}') + 1]
    # Balance braces
    while snippet.count('{') != snippet.count('}'):
        snippet = snippet[:snippet.rfind('}')]
    lines = snippet.split('<ST>')
    return ''.join(lines), startline, startline + len(lines) - 1, s_idx + len(lines) - 1

# ---------- Project processing ----------
def process_project(project_name: str, cfg: dict) -> None:
    """Process each project version: extract Java code and update metrics DataFrame."""
    for src_path, metrics_path in zip(cfg['sources'], cfg['metrics']):
        src_root = Path(src_path)
        csv_file = Path(metrics_path)
        if not src_root.exists() or not csv_file.exists():
            logger.error(f"Path not found: {src_root} or {csv_file}")
            continue
        logger.info(f"Processing {project_name} - {src_root.name}")
        df = load_metrics(csv_file)
        for java_file in find_java_files(src_root):
            code = java_file.read_text(encoding='utf-8')
            classes = extract_classes_from_code(code, code.splitlines())
            for fqn, src in classes.items():
                if fqn in df.index:
                    df.at[fqn, 'source'] = src
        save_results(df, csv_file)

# ---------- Main entry point ----------
def main(config_file: Path) -> None:
    """Load configuration and process each project."""
    cfg = load_config(config_file)
    for project, paths in cfg['projects'].items():
        process_project(project, paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract full Java code (including headers) from PROMISE projects'
    )
    parser.add_argument(
        '--config', type=Path, default=Path('config.yaml'),
        help='Path to the YAML configuration file'
    )
    args = parser.parse_args()
    main(args.config)