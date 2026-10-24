import json
from pathlib import Path
import sys
import pytest
import polars as pl
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from data_contracts.schema_golden_set import BusinessEntity, ResolvedMatch, ProspectScore
from mx_osint.config import Settings


def test_scian_chronotype_mapping_structure():
    path = Path("data_contracts/scian_chronotype_mapping.json")
    assert path.exists(), "scian_chronotype_mapping.json must exist"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    # Every key should map to chronotype and gatekeeper_roles
    for k, v in data.items():
        assert "chronotype" in v
        assert "gatekeeper_roles" in v
        assert isinstance(v["gatekeeper_roles"], list)


def test_scian_b2b_propensity_structure():
    path = Path("data_contracts/scian_b2b_propensity.json")
    assert path.exists(), "scian_b2b_propensity.json must exist"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    for k, v in data.items():
        assert "weight" in v
        assert isinstance(v["weight"], (int, float))
        assert 0.0 <= v["weight"] <= 100.0


def test_schema_golden_set_models():
    # Test BusinessEntity
    entity_data = {
        "id": "12345",
        "clee": "1102000001",
        "nombre_estab": "TECNOLOGIA Y SOFTWARE SA DE CV",
        "raz_social": "TECNOLOGIA Y SOFTWARE SA DE CV",
        "codigo_act": "541512",
        "nombre_act": "Desarrollo de software",
        "per_ocu": "31 a 50 personas",
        "telefono": "4771234567",
        "correoelec": "contacto@techcorp.mx",
        "www": "https://techcorp.mx",
        "lat": 21.1221,
        "lon": -101.6826,
        "ent": "11",
        "mun": "020",
        "fecha_alta": "2020-01"
    }
    entity = BusinessEntity(**entity_data)
    assert entity.id == "12345"
    assert entity.lat == 21.1221

    # Test ResolvedMatch
    match_data = {
        "entity_id_a": "ent_001",
        "entity_id_b": "ent_002",
        "match_score": 0.95,
        "is_match": True,
        "canonical_entity_id": "ent_001"
    }
    match = ResolvedMatch(**match_data)
    assert match.match_score == 0.95

    # Test ProspectScore
    prospect_data = {
        "entity_id": "ent_001",
        "nombre_estab": "TECNOLOGIA Y SOFTWARE SA DE CV",
        "b2b_score": 88.5,
        "channel": "Email_RH_Compras",
        "digital_maturity_score": 3,
        "chronotype": "Office_Standard_8AM"
    }
    prospect = ProspectScore(**prospect_data)
    assert prospect.b2b_score == 88.5


def test_synthetic_fixtures():
    denue_path = Path("fixtures/synthetic_denue_sample.parquet")
    assert denue_path.exists(), "synthetic_denue_sample.parquet must exist"
    df = pl.read_parquet(denue_path)
    assert len(df) == 50
    assert "nombre_estab" in df.columns
    assert "per_ocu" in df.columns
    assert "lat" in df.columns

    osm_path = Path("fixtures/synthetic_osm_sample.geojson")
    assert osm_path.exists(), "synthetic_osm_sample.geojson must exist"
    with open(osm_path, "r", encoding="utf-8") as f:
        osm_data = json.load(f)
    assert osm_data.get("type") == "FeatureCollection"
    assert len(osm_data.get("features", [])) == 20


def test_config_settings():
    settings = Settings(
        INEGI_STATE_CODE="11",
        TARGET_LAT=21.1221,
        TARGET_LON=-101.6826,
        RADIUS_KM=3.0,
        MIN_EMPLOYEES_STRATA="11 a 30 personas",
        DDG_SLEEP_SEC=2.0
    )
    assert settings.INEGI_STATE_CODE == "11"
    assert settings.RADIUS_KM == 3.0
