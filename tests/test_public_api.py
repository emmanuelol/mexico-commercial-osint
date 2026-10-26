import pytest
import polars as pl
from pathlib import Path

from mx_osint import (
    DENUELoader,
    OSMLoader,
    SpatialBlocker,
    ChronotypeImputer,
    B2BPropensityEnricher,
    EmpiricalAdaBoostScorer,
    ChannelClassifier,
    normalize_mexican_business_name,
    normalize_mexican_legal_name,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_public_exports_exist():
    assert callable(DENUELoader)
    assert callable(OSMLoader)
    assert callable(SpatialBlocker)
    assert callable(ChronotypeImputer)
    assert callable(B2BPropensityEnricher)
    assert callable(EmpiricalAdaBoostScorer)
    assert callable(ChannelClassifier)
    assert callable(normalize_mexican_business_name)
    assert callable(normalize_mexican_legal_name)


def test_normalize_mexican_business_name_canonical():
    raw_denue = "DISTRIBUIDORA DE INSUMOS DEL BAJÍO, S.A. DE C.V."
    raw_osm = "Distribuidora de Insumos del Bajio"

    norm_denue = normalize_mexican_business_name(raw_denue)
    norm_osm = normalize_mexican_business_name(raw_osm)

    assert norm_denue == "distribuidora insumos bajio"
    assert norm_osm == "distribuidora insumos bajio"
    assert norm_denue == norm_osm


def test_chronotype_imputer_and_b2b_propensity():
    imputer = ChronotypeImputer()
    enricher = B2BPropensityEnricher()

    profile = imputer.impute(scian_code="311811")
    assert "chronotype_key" in profile
    assert profile["chronotype_key"] == "Turno_Ingenieria_630_730AM"

    propensity = enricher.evaluate(scian_code="311811", estrato_personal="31 a 50 personas")
    assert "b2b_propensity_score" in propensity
    assert "est_monthly_coffee_rev_mxn" in propensity
    assert propensity["b2b_propensity_score"] > 0


def test_loaders_and_spatial_blocker():
    denue_loader = DENUELoader()
    osm_loader = OSMLoader()

    denue_df = denue_loader.load(FIXTURES_DIR / "synthetic_denue_sample.parquet")
    assert isinstance(denue_df, pl.DataFrame)
    assert len(denue_df) == 50

    osm_df = osm_loader.load(FIXTURES_DIR / "synthetic_osm_sample.geojson")
    assert isinstance(osm_df, pl.DataFrame)
    assert len(osm_df) > 0
    assert "lat" in osm_df.columns
    assert "lon" in osm_df.columns

    blocker = SpatialBlocker(max_dist_m=100.0)
    joined_df = blocker.block(denue_df, osm_df)
    assert isinstance(joined_df, pl.DataFrame)
    assert "nearest_osm_id" in joined_df.columns
    assert "osm_match_dist_m" in joined_df.columns


def test_channel_classifier():
    classifier = ChannelClassifier()
    account = {
        "canonical_parent_name": "CORPORATIVO FINANCIERO BAJIO",
        "scian_code": "522110",
        "email": "contacto@bajiofinanzas.mx",
        "distance_km": 1.2,
        "total_est_employees_midpoint": 40,
    }
    result = classifier.classify(account)
    assert "best_contact_channel" in result
    assert "playbook_code" in result
    assert result["best_contact_channel"] == "Direct Email (Verified DENUE)"
