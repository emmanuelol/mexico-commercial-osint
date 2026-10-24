# Notebook to Module Migration Architecture

## Philosophy

Notebooks are **validation environments**, not long-term production modules. Every production function begins in a notebook cell, is tested and validated with realistic data, and contains a migration pointer:

```python
# MIGRATES TO: src/mx_osint/<subpackage>/<module>.py::<function_name>
```

## Migration Workflow

1. Prototype and validate the logic inside `notebooks/`.
2. Ensure every function is:
   - Self-contained (no global mutable variables)
   - Fully type-hinted
   - Documented with clear docstrings
3. Execute `python scripts/migrate_cells.py --notebooks notebooks/ --target src/mx_osint/`.
4. Run regression tests via `pytest tests/`.
