from pathlib import Path
import json
import polars as pl

def export_dual(df_canonical: pl.DataFrame, df_field_routed: pl.DataFrame, base_dirs: list = None) -> dict:
    """Exports private CRM tables and public zero-PII aggregates."""
    if base_dirs is None:
        base_dirs = [Path("data"), Path("../data"), Path("notebooks/data")]
    
    top30_export = None
    email_ready = None
    pub_payload = None

    # Cell 3.6: Strict Clean-Room Dual Export & CRM Ready Files
    
    for base_dir in [Path("data"), Path("../data"), Path("notebooks/data")]:
        if base_dir.exists():
            priv_dir = base_dir / "private"
            proc_dir = base_dir / "processed"
            priv_dir.mkdir(parents=True, exist_ok=True)
            proc_dir.mkdir(parents=True, exist_ok=True)
    
            # 1. Private Top-30 B2B Prospects
            top30_export = df_canonical.head(30)
            top30_export.write_csv(priv_dir / "top_30_b2b_prospects.csv")
    
            # 2. Private Verified Low-Risk Email Sequences
            email_ready = df_canonical.filter((pl.col("contact_risk") == "low") & (pl.col("email_confidence").is_in(["high", "medium-high"])))
            email_ready.write_csv(priv_dir / "stage3_email_sequence_ready.csv")
    
            # 3. Private Field Sales Routes
            df_field_routed.write_csv(priv_dir / "stage3_field_sales_routes.csv")
    
            # 4. Public Clean-Room Aggregate Index (Zero PII)
            pub_agg = (
                df_canonical
                .group_by(["chronotype_key", "b2b_offer_playbook_code"])
                .agg([
                    pl.len().alias("total_accounts"),
                    pl.col("site_count").sum().alias("total_physical_sites"),
                    pl.col("total_est_employees_midpoint").sum().alias("total_headcount"),
                    pl.col("consolidated_payroll_mxn").sum().alias("total_payroll_mxn"),
                    pl.col("est_monthly_coffee_rev_mxn").sum().alias("total_est_monthly_coffee_rev_mxn"),
                    pl.col("b2b_propensity_score").mean().alias("avg_propensity_score"),
                ])
                .sort("total_payroll_mxn", descending=True)
            )
            pub_payload = {
                "meta": {
                    "generator": "mexico-commercial-osint",
                    "stage": "03_b2b_propensity_activation",
                    "base_anchor": "The Coffee — Bajío Hub (GTO)",
                    "total_canonical_accounts": len(df_canonical),
                    "total_actionable_sites": len(df_field_routed),
                    "total_monthly_coffee_opportunity_mxn": float(df_canonical["est_monthly_coffee_rev_mxn"].sum()),
                },
                "aggregates": pub_agg.to_dicts(),
            }
            with open(proc_dir / "public_b2b_aggregates.json", "w", encoding="utf-8") as f:
                json.dump(pub_payload, f, indent=2, ensure_ascii=False)
    

    return {
        "top30_count": len(top30_export) if top30_export is not None else 0,
        "email_ready_count": len(email_ready) if email_ready is not None else 0,
        "routed_sites_count": len(df_field_routed),
        "public_payload": pub_payload,
    }
