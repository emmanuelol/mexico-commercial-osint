#!/usr/bin/env python3
"""
Migrates cell functions from Jupyter Notebooks to src/mx_osint/ modules based on
'# MIGRATES TO: <target_path>::<symbol_name>' comments.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

MIGRATE_PATTERN = re.compile(r"#\s*MIGRATES\s*TO:\s*([^\s:]+)::([^\s]+)")

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
    for item in all_cells:
        target_file = Path(item["target_path"])
        print(f"  [{item['notebook']}] -> {item['target_path']} :: {item['symbol']}")
        if not args.dry_run:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            init_file = target_file.parent / "__init__.py"
            if not init_file.exists():
                init_file.touch()

            if target_file.exists() and not args.force:
                print(f"    Target {target_file} exists. Skipping (use --force to overwrite).")
                continue

            with open(target_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n# Migrated from {item['notebook']} symbol {item['symbol']}\n")
                f.write(item["source"] + "\n")
            print(f"    Appended to {target_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
