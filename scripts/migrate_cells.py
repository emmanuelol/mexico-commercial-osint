#!/usr/bin/env python3
"""
Migrates cell functions from Jupyter Notebooks to src/mx_osint/ modules based on
'# MIGRATES TO: <target_path>::<symbol_name>' comments.
Uses AST extraction to ensure clean, importable Python modules free of interactive notebook code.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

MIGRATE_PATTERN = re.compile(r"#\s*MIGRATES\s*TO:\s*([^\s:]+)::([^\s]+)")

# Variables/expressions representing interactive notebook execution to exclude
EXCLUDE_ASSIGN_TARGETS = {
    "df", "raw_df", "df_prep", "df_sites", "df_stage1", "df_canonical", "df_field_routed",
    "top20", "top30_export", "top_routes", "top_accounts", "email_ready",
    "m_act", "pub_agg", "pub_payload", "settings", "raw_zip", "records", "n",
    "parent_map", "clusters", "lats", "lons", "emvs", "eval_rows", "t4_rows", "t5_rows",
    "r1", "r2", "r3", "sites_with_emv", "route_names", "playbook_results", "emv_results",
    "aumovio_1", "aumovio_2", "pape"
}


def extract_cells_from_notebook(notebook_path: Path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    extracted = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        match = MIGRATE_PATTERN.search(source)
        if match:
            target_rel_path = match.group(1).strip()
            symbol_name = match.group(2).strip()
            extracted.append({
                "notebook": notebook_path.name,
                "target_path": target_rel_path,
                "symbol": symbol_name,
                "source": source
            })
    return extracted


def clean_cell_source_with_ast(source: str, symbol_name: str) -> str:
    """Extracts only imports, constants, classes, and functions from cell source."""
    # Special wrap for prospect_report export_dual if not already a FunctionDef
    if symbol_name == "export_dual" and "def export_dual" not in source:
        clean_lines = []
        for line in source.splitlines():
            if line.startswith("# MIGRATES TO:"):
                continue
            if line.startswith("print("):
                continue
            clean_lines.append("    " + line)
        body = "\n".join(clean_lines)
        return f'''from pathlib import Path
import json
import polars as pl

def export_dual(df_canonical: pl.DataFrame, df_field_routed: pl.DataFrame, base_dirs: list = None) -> dict:
    """Exports private CRM tables and public zero-PII aggregates."""
    if base_dirs is None:
        base_dirs = [Path("data"), Path("../data"), Path("notebooks/data")]
    
    top30_export = None
    email_ready = None
    pub_payload = None

{body}

    return {{
        "top30_count": len(top30_export) if top30_export is not None else 0,
        "email_ready_count": len(email_ready) if email_ready is not None else 0,
        "routed_sites_count": len(df_field_routed),
        "public_payload": pub_payload,
    }}
'''

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    clean_nodes = []
    def _get_target_names(t):
        if isinstance(t, ast.Name):
            return [t.id]
        elif isinstance(t, (ast.Tuple, ast.List)):
            names = []
            for elt in t.elts:
                names.extend(_get_target_names(elt))
            return names
        return []

    for node in tree.body:
        # Keep all imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            clean_nodes.append(node)
        # Keep class and function definitions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            clean_nodes.append(node)
        # Keep module-level constant assignments and type annotations
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_names = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    target_names.extend(_get_target_names(t))
            elif isinstance(node, ast.AnnAssign):
                target_names.extend(_get_target_names(node.target))

            # Only keep module-level constants (PEP8 uppercase or dunder/private)
            if target_names and all(t.isupper() or t.startswith("__") for t in target_names):
                clean_nodes.append(node)
        # Exclude bare expressions (print, display, loops over notebook datasets)

    if not clean_nodes:
        return source

    tree.body = clean_nodes
    unparsed = ast.unparse(tree)

    missing_imports = []
    if "pl." in unparsed and "import polars" not in unparsed:
        missing_imports.append("import polars as pl")
    if "np." in unparsed and "import numpy" not in unparsed:
        missing_imports.append("import numpy as np")
    if "Path(" in unparsed and "from pathlib import Path" not in unparsed and "import Path" not in unparsed:
        missing_imports.append("from pathlib import Path")
    if "math." in unparsed and "import math" not in unparsed:
        missing_imports.append("import math")
    if "json." in unparsed and "import json" not in unparsed:
        missing_imports.append("import json")
    if "re." in unparsed and "import re" not in unparsed:
        missing_imports.append("import re")
    if "logger." in unparsed and "logger" not in unparsed:
        missing_imports.append("from loguru import logger")
    if "time." in unparsed and "import time" not in unparsed:
        missing_imports.append("import time")
    if "requests." in unparsed and "import requests" not in unparsed:
        missing_imports.append("import requests")
    if "settings." in unparsed and "Settings" not in unparsed:
        missing_imports.append("from mx_osint.config import Settings\nsettings = Settings()")
    if ("ANCHOR_LAT" in unparsed or "ANCHOR_LON" in unparsed) and not re.search(r"^\s*ANCHOR_LAT\b", unparsed, re.MULTILINE):
        missing_imports.append(
            "from mx_osint.config import Settings as _Settings\n"
            "_s = _Settings()\n"
            "ANCHOR_LAT, ANCHOR_LON = _s.TARGET_LAT, _s.TARGET_LON"
        )
    if "STRATA_ORDER" in unparsed and "from mx_osint.models.denue" not in unparsed and not re.search(r"^\s*STRATA_ORDER\s*[:=]", unparsed, re.MULTILINE):
        missing_imports.append("from mx_osint.models.denue import STRATA_ORDER, STRATA_HEADCOUNT_MIDPOINT")
    elif "STRATA_HEADCOUNT_MIDPOINT" in unparsed and "from mx_osint.models.denue" not in unparsed and not re.search(r"^\s*STRATA_HEADCOUNT_MIDPOINT\s*[:=]", unparsed, re.MULTILINE):
        missing_imports.append("from mx_osint.models.denue import STRATA_HEADCOUNT_MIDPOINT")
    if "STAGE2_CONFIG" in unparsed and not re.search(r"^\s*STAGE2_CONFIG\s*[:=]", unparsed, re.MULTILINE):
        missing_imports.append(
            'STAGE2_CONFIG = {"osm_match_radius_m": 50.0, "fuzzy_threshold": 0.85, '
            '"partial_upgrade_score_threshold": 4.8, "min_headcount_corporate_partial": 31}'
        )
    if "ddg_cache" in unparsed and not re.search(r"^\s*ddg_cache\s*[:=]", unparsed, re.MULTILINE):
        missing_imports.append("ddg_cache: dict[str, dict] = {}")

    if missing_imports:
        unparsed = "\n".join(missing_imports) + "\n\n" + unparsed

    return unparsed


def main():
    parser = argparse.ArgumentParser(description="Migrate notebook cells to production modules.")
    parser.add_argument("--notebooks", type=str, default="notebooks/", help="Directory containing .ipynb files")
    parser.add_argument("--target", type=str, default="src/mx_osint/", help="Base destination directory")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing destination files")
    parser.add_argument("--dry-run", action="store_true", help="Print migration operations without writing files")

    args = parser.parse_args()
    nb_dir = Path(args.notebooks)
    if not nb_dir.exists():
        print(f"Directory {nb_dir} does not exist.")
        return 0

    notebooks = sorted(list(nb_dir.glob("*.ipynb")))
    if not notebooks:
        print(f"No notebooks found in {nb_dir}.")
        return 0

    all_cells = []
    for nb_file in notebooks:
        all_cells.extend(extract_cells_from_notebook(nb_file))

    print(f"Found {len(all_cells)} migratable cell(s).")
    touched_files = set()

    for item in all_cells:
        target_file = Path(item["target_path"])
        print(f"  [{item['notebook']}] -> {item['target_path']} :: {item['symbol']}")
        if not args.dry_run:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            init_file = target_file.parent / "__init__.py"
            if not init_file.exists():
                init_file.touch()

            # For config.py, preserve existing clean canonical implementation
            if target_file == Path("src/mx_osint/config.py") and target_file.exists():
                print(f"    Preserving canonical {target_file}")
                continue

            cleaned_code = clean_cell_source_with_ast(item["source"], item["symbol"])

            if target_file not in touched_files:
                mode = "w" if (args.force or not target_file.exists()) else "a"
                touched_files.add(target_file)
            else:
                mode = "a"

            # If overwriting with identical content, leave file untouched
            if mode == "w" and target_file.exists():
                with open(target_file, "r", encoding="utf-8") as f_ex:
                    if f_ex.read().strip() == cleaned_code.strip():
                        print(f"    Unchanged: {target_file}")
                        continue

            with open(target_file, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write(f"\n\n# Migrated from {item['notebook']} symbol {item['symbol']}\n")
                f.write(cleaned_code.strip() + "\n")
            print(f"    {'Overwrote' if mode == 'w' else 'Appended to'} {target_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
