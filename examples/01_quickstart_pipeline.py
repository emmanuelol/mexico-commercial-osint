#!/usr/bin/env python3
"""
01_quickstart_pipeline.py
End-to-End OSINT Pipeline: Ingestion -> Spatial Resolution -> Imputation -> Activation.
Runs locally in under 3 seconds using privacy-safe synthetic fixtures.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_contracts.schema_golden_set import BusinessEntity
from mx_osint import (
    DENUELoader, OSMLoader, SpatialBlocker,
    ChronotypeImputer, B2BPropensityEnricher, ChannelClassifier,
    normalize_mexican_business_name
)

ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    # 1. Ingestion & Golden Set Contract Validation
    denue_df = DENUELoader().load(ROOT / "fixtures" / "synthetic_denue_sample.parquet")
    osm_df = OSMLoader().load(ROOT / "fixtures" / "synthetic_osm_sample.geojson")
    BusinessEntity(**denue_df.to_dicts()[0])  # Enforce schema golden set

    # 2. Entity Resolution & Spatial Candidate Blocking
    resolved = SpatialBlocker(max_dist_m=120.0).block(denue_df, osm_df)

    # 3. Enrichment, Prior-Driven Imputation & Channel Activation
    imputer, enricher, classifier = ChronotypeImputer(), B2BPropensityEnricher(), ChannelClassifier()
    prospects = []
    for r in resolved.iter_rows(named=True):
        norm_name = normalize_mexican_business_name(r["nombre_estab"])
        chrono = imputer.impute(r["codigo_act"])
        propensity = enricher.evaluate(
            scian_code=r["codigo_act"], estrato_personal=r["per_ocu"],
            chronotype_key=chrono["chronotype_key"], distance_km=0.8
        )
        action = classifier.classify({**r, "scian_code": r["codigo_act"], "email": r.get("correoelec")})
        prospects.append({
            "name": norm_name[:26],
            "scian": r["codigo_act"],
            "window": chrono["time_window"],
            "score": propensity["b2b_propensity_score"],
            "channel": action["best_contact_channel"][:22]
        })

    # 4. Display Ranked B2B Prospect Intelligence Summary
    prospects.sort(key=lambda x: x["score"], reverse=True)
    print("\n🇲🇽 MEXICO COMMERCIAL OSINT — END-TO-END PIPELINE SUMMARY")
    print(f"{'RANK':<5} | {'BUSINESS NAME':<26} | {'SCIAN':<6} | {'WINDOW':<16} | {'SCORE':<5} | {'ACTION CHANNEL':<22}")
    print("-" * 92)
    for idx, p in enumerate(prospects[:10], start=1):
        print(f"{idx:<5} | {p['name']:<26} | {p['scian']:<6} | {p['window']:<16} | {p['score']:<5} | {p['channel']:<22}")
    print(f"\n✅ Pipeline executed successfully against {len(denue_df)} DENUE records & {len(osm_df)} OSM POIs.\n")

if __name__ == "__main__":
    main()
