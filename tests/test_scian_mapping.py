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
    assert hasattr(settings, "ANCHOR_LABEL")
    assert "Bajío Hub" in settings.ANCHOR_LABEL
    assert hasattr(settings, "PROCESSED_DIR")


def test_activation_contracts():
    from mx_osint.activation.contracts import (
        CouponSpec,
        OutreachPayload,
        DryRunPromoProvider,
        DryRunOutreachDispatcher,
    )

    # Valid coupon
    spec = CouponSpec(
        code="VECINO15",
        coupon_type="discount",
        description="local_partners",
        discount_pct=15.0,
        store_id=525,
    )
    assert spec.code == "VECINO15"
    assert spec.discount_pct == 15.0

    # Provider dry run
    provider = DryRunPromoProvider(max_discount_pct=15.0)
    res = provider.provision_coupon(spec)
    assert res["status"] == "PROVISIONED_DRY_RUN"
    assert provider.verify_coupon("VECINO15") is True
    assert provider.verify_coupon("UNKNOWN") is False

    # Dispatcher dry run
    dispatcher = DryRunOutreachDispatcher()
    payload = OutreachPayload(
        canonical_entity_id="ent_test",
        recipient_email="contacto@empresa.com",
        email_subject="Convenio Corporativo",
        email_body="Cuerpo del mensaje",
        chronotype_key="Ejecutivo_Oficina_800_930AM",
        coupon_code="VECINO15",
        contact_risk="low",
        email_confidence="high",
    )
    d_res = dispatcher.dispatch(payload)
    assert d_res["status"] == "DISPATCHED_DRY_RUN"
    assert len(dispatcher.dispatched) == 1

    # Blocked by guardrail
    blocked_payload = OutreachPayload(
        canonical_entity_id="ent_test2",
        recipient_email="contacto@sintetizado.com",
        email_subject="Test",
        email_body="Test",
        chronotype_key="Turno_630AM",
        contact_risk="high",
        email_confidence="low",
    )
    with pytest.raises(ValueError, match="Outreach blocked by guardrail"):
        dispatcher.dispatch(blocked_payload)


def test_migrated_modules_imports():
    import mx_osint.models.denue
    import mx_osint.sources.denue_loader
    import mx_osint.sources.osm_loader
    import mx_osint.scoring.b2b_qualifier
    import mx_osint.scoring.empirical_adaboost_scorer
    import mx_osint.resolution.spatial_blocker
    import mx_osint.resolution.mex_name_normalizer
    import mx_osint.resolution.contact_enricher
    import mx_osint.enrichment.chronotype_imputer
    import mx_osint.enrichment.digital_maturity
    import mx_osint.enrichment.b2b_propensity
    import mx_osint.activation.channel_classifier
    import mx_osint.reporting.prospect_report

    assert hasattr(mx_osint.models.denue, "DENUEBusiness")
    assert hasattr(mx_osint.sources.denue_loader, "read_denue_csv")
    assert hasattr(mx_osint.enrichment.b2b_propensity, "calculate_b2b_emv_record")
    assert hasattr(mx_osint.reporting.prospect_report, "export_dual")


def test_migrated_standalone_callability():
    from mx_osint.sources.denue_loader import filter_by_strata
    from mx_osint.enrichment.digital_maturity import classify_digital_maturity_record, discover_channel_ddg
    from mx_osint.activation.channel_classifier import assign_offer_playbook, ANCHOR_LAT, ANCHOR_LON
    from mx_osint.models.denue import generate_synthetic_denue_fixture

    # Test standalone filter_by_strata without NameError on STRATA_ORDER
    sample_df = pl.DataFrame({"employee_strata": ["11 a 30 personas", "0 a 5 personas"]})
    res_df = filter_by_strata(sample_df, "11 a 30 personas")
    assert len(res_df) == 1

    # Test standalone discover_channel_ddg without NameError on ddg_cache
    res_ddg = discover_channel_ddg("Empresa Test")
    assert "status" in res_ddg

    # Test standalone classify_digital_maturity_record
    mat = classify_digital_maturity_record({"effective_contact": "test@corp.mx", "email_confidence": "high"})
    assert "digital_maturity_score" in mat

    # Test ANCHOR_LAT and ANCHOR_LON in channel_classifier
    assert round(ANCHOR_LAT, 2) == 21.12
    assert round(ANCHOR_LON, 2) == -101.68
    pb = assign_offer_playbook({"canonical_parent_name": "Test Plant", "scian_code": "3361"})
    assert "playbook_code" in pb

    # Test docstring clean-room
    doc = generate_synthetic_denue_fixture.__doc__ or ""
    assert "Paseo MS" not in doc
    assert "target anchor" in doc

    # Test standalone contact_enricher and empirical_adaboost_scorer constants
    import mx_osint.resolution.contact_enricher as ce
    assert hasattr(ce, "STAGE2_CONFIG")
    assert "partial_upgrade_score_threshold" in ce.STAGE2_CONFIG

    import mx_osint.scoring.empirical_adaboost_scorer as eas
    assert hasattr(eas, "STRATA_HEADCOUNT_MIDPOINT")
    assert "11 a 30 personas" in eas.STRATA_HEADCOUNT_MIDPOINT


