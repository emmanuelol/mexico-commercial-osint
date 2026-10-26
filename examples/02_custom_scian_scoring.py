#!/usr/bin/env python3
"""
02_custom_scian_scoring.py
Extending mx_osint for Custom B2B Verticals (e.g., Fintech, Logistics, SaaS).
Demonstrates custom SCIAN contract weighting and propensity re-calibration.
"""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mx_osint import DENUELoader, ChronotypeImputer, B2BPropensityEnricher

def main() -> None:
    # 1. Load Synthetic DENUE Sample
    denue_loader = DENUELoader()
    df = denue_loader.load(ROOT / "fixtures" / "synthetic_denue_sample.parquet")

    # 2. Define Custom Domain Weights (Fintech & Corporate SaaS Underwriting)
    # Developers can override base contracts with vertical-specific business hypotheses
    custom_vertical_weights = {
        "51": {"sector": "Información y Software", "weight": 95.0, "vertical_multiplier": 1.4},
        "52": {"sector": "Finanzas y Seguros", "weight": 98.0, "vertical_multiplier": 1.5},
        "54": {"sector": "Consultoría y Servicios Profesionales", "weight": 90.0, "vertical_multiplier": 1.3},
        "33": {"sector": "Manufactura Automotriz & Alta Tecnología", "weight": 70.0, "vertical_multiplier": 1.1},
        "46": {"sector": "Comercio al por menor tradicional", "weight": 25.0, "vertical_multiplier": 0.6},
    }

    # 3. Instantiate Standard Enrichers
    imputer = ChronotypeImputer()
    base_enricher = B2BPropensityEnricher()

    print("\n🔍 EVALUATING CUSTOM VERTICAL SCORING: FINTECH / SAAS B2B READINESS")
    print(f"{'BUSINESS NAME':<28} | {'SCIAN':<6} | {'BASE EMV':<10} | {'CUSTOM FIT':<10} | {'ADJUSTED SCORE':<14}")
    print("-" * 80)

    for row in df.head(8).iter_rows(named=True):
        code = str(row["codigo_act"])
        prefix2 = code[:2]
        chrono = imputer.impute(code)
        base_res = base_enricher.evaluate(
            scian_code=code,
            estrato_personal=row["per_ocu"],
            chronotype_key=chrono["chronotype_key"]
        )

        # Apply domain vertical multiplier
        multiplier = custom_vertical_weights.get(prefix2, {}).get("vertical_multiplier", 1.0)
        custom_score = min(100.0, round(base_res["b2b_propensity_score"] * multiplier, 1))

        print(
            f"{row['nombre_estab'][:28]:<28} | {code:<6} | "
            f"${base_res['est_monthly_coffee_rev_mxn']:>8.2f} | "
            f"{multiplier:>9.2f}x | {custom_score:>13.1f}"
        )

    print("\n💡 Custom domain scoring completed successfully.")
    print("   To persist custom vertical weights, save as JSON and pass to B2BPropensityEnricher(weights_path=...).\n")

if __name__ == "__main__":
    main()
