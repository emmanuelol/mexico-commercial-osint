import math
import re
import unicodedata
import hashlib
from pathlib import Path
import polars as pl
import pytest
from rapidfuzz import fuzz, process

ROOT_DIR = Path(__file__).resolve().parent.parent

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b("
    r"S\.?\s*A\.?\s*P\.?\s*I\.?\s*(?:DE\s*C\.?\s*V\.?)?|"
    r"S\.?\s*A\.?\s*B\.?\s*(?:DE\s*C\.?\s*V\.?)?|"
    r"S\.?\s*A\.?\s*DE\s*C\.?\s*V\.?|"
    r"S\.?\s*DE\s*R\.?\s*L\.?\s*(?:DE\s*C\.?\s*V\.?)?|"
    r"S\.?\s*C\.?\s*DE\s*R\.?\s*L\.?|"
    r"S\.?\s*C\.?|"
    r"A\.?\s*C\.?|"
    r"S\.?\s*A\.?|"
    r"SAB\s*DE\s*CV"
    r")\b",
    re.IGNORECASE
)

BRANCH_QUALIFIER_REGEX = re.compile(
    r"\b(PLANTA|MATRIZ|SUCURSAL|CORPORATIVO|CAMPUS|NORTE|LEON|GUANAJUATO|BAJIO|MEXICO)\b",
    re.IGNORECASE
)

PUNCTUATION_REGEX = re.compile(r"[,.\-'\"]")


def normalize_mexican_legal_name(name: str) -> str:
    """Normalizes a Mexican company or trade name: uppercase, accent removal, legal suffix stripping."""
    if not name or not isinstance(name, str):
        return ""
    # 1. Strip accents via NFKD
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).upper()
    
    # 2. Strip legal suffixes
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)
    
    # 3. Strip punctuation
    text = PUNCTUATION_REGEX.sub(" ", text)
    
    # 4. Collapse whitespace
    text = " ".join(text.split()).strip()
    return text


def strip_corporate_branch_qualifiers(name: str) -> str:
    """Strips secondary branch/plant descriptors (e.g., PLANTA, MATRIZ, NORTE) for campus parent resolution."""
    cleaned = normalize_mexican_legal_name(name)
    cleaned = BRANCH_QUALIFIER_REGEX.sub(" ", cleaned)
    cleaned = PUNCTUATION_REGEX.sub(" ", cleaned)
    return " ".join(cleaned.split()).strip()


def encode_geohash_manual(latitude: float, longitude: float, precision: int = 6) -> str:
    """Vector-friendly standard geohash encoder in pure Python without external C libraries."""
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2.0
            if longitude > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2.0
            if latitude > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(base32[ch])
            bit = 0
            ch = 0
    return "".join(geohash)


def assign_geohash_block(df: pl.DataFrame, precision: int = 6) -> pl.DataFrame:
    geohashes = [
        encode_geohash_manual(lat, lon, precision=precision)
        for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list())
    ]
    return df.with_columns(pl.Series("geohash", geohashes))


def assign_entity_ids(df: pl.DataFrame) -> pl.DataFrame:
    entity_ids = []
    for norm_name, gh in zip(df["norm_name"].to_list(), df["geohash"].to_list()):
        raw_key = f"{norm_name}_{gh[:4]}".encode("utf-8")
        entity_ids.append(f"ent_{hashlib.sha1(raw_key).hexdigest()[:12]}")
    return df.with_columns(pl.Series("entity_id", entity_ids))


def spatial_join_osm_context(
    denue_df: pl.DataFrame,
    osm_df: pl.DataFrame,
    max_dist_m: float = 50.0,
) -> pl.DataFrame:
    """Cross-source spatial resolution between DENUE and OSM amenities respecting ODbL boundaries."""
    if len(osm_df) == 0:
        return denue_df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("nearest_osm_id"),
            pl.lit(None).cast(pl.Float64).alias("osm_match_dist_m"),
            pl.lit(False).alias("osm_verified_storefront"),
            pl.lit("").alias("osm_poi_context"),
        ])

    osm_records = osm_df.select(["osm_id", "lat", "lon", "name", "amenity", "office", "shop"]).to_dicts()
    r_earth_m = 6371000.0
    rad = math.pi / 180.0

    nearest_ids = []
    nearest_dists = []
    verified_flags = []
    contexts = []

    for row in denue_df.iter_rows(named=True):
        d_lat, d_lon = row["lat"], row["lon"]
        best_dist = float("inf")
        best_osm = None

        for osm in osm_records:
            o_lat, o_lon = osm["lat"], osm["lon"]
            dlat = (o_lat - d_lat) * rad
            dlon = (o_lon - d_lon) * rad
            a = math.sin(dlat / 2.0) ** 2 + math.cos(d_lat * rad) * math.cos(o_lat * rad) * math.sin(dlon / 2.0) ** 2
            dist_m = 2.0 * math.asin(math.sqrt(a)) * r_earth_m

            if dist_m < best_dist:
                best_dist = dist_m
                best_osm = osm

        if best_osm and best_dist <= 250.0:
            nearest_ids.append(str(best_osm["osm_id"]))
            nearest_dists.append(round(best_dist, 1))
            verified_flags.append(best_dist <= max_dist_m)
            ctx = best_osm.get("name") or best_osm.get("amenity") or best_osm.get("office") or best_osm.get("shop") or ""
            contexts.append(str(ctx))
        else:
            nearest_ids.append(None)
            nearest_dists.append(None)
            verified_flags.append(False)
            contexts.append("")

    return denue_df.with_columns([
        pl.Series("nearest_osm_id", nearest_ids, dtype=pl.Utf8),
        pl.Series("osm_match_dist_m", nearest_dists, dtype=pl.Float64),
        pl.Series("osm_verified_storefront", verified_flags, dtype=pl.Boolean),
        pl.Series("osm_poi_context", contexts, dtype=pl.Utf8),
    ])


def classify_b2b_email_role(email: str | None) -> str:
    """Classifies B2B email into decision maker, functional generic inbox, or personal/free email."""
    if not email or not isinstance(email, str) or "@" not in email:
        return "none"
    clean_em = email.strip().lower()
    free_domains = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "live.com", "prodigy.net.mx"]
    if any(clean_em.endswith(f"@{d}") for d in free_domains):
        return "free_personal"
    
    user_part = clean_em.split("@")[0]
    decision_makers = ["director", "gerente", "rh", "recursoshumanos", "purchasing", "compras", "talento", "ceo", "cfo"]
    generic_roles = ["contacto", "info", "recepcion", "atencion", "ventas", "servicio", "facturacion"]
    
    if any(dm in user_part for dm in decision_makers):
        return "decision_maker_role"
    if any(gen in user_part for gen in generic_roles):
        return "generic_role_inbox"
    return "corporate_direct"


def test_normalize_mexican_legal_suffixes():
    test_cases = [
        ("GRUPO INDUSTRIAL DEL BAJÍO S.A. DE C.V.", "GRUPO INDUSTRIAL DEL BAJIO"),
        ("TECNOLOGIA Y SOFTWARE S.A.P.I. DE C.V.", "TECNOLOGIA Y SOFTWARE"),
        ("LOGISTICA NACIONAL S. DE R.L. DE C.V.", "LOGISTICA NACIONAL"),
        ("SERVICIOS CONTABLES Y FISCALES S.C.", "SERVICIOS CONTABLES Y FISCALES"),
        ("FUNDACION COMUNITARIA A.C.", "FUNDACION COMUNITARIA"),
        ("COMERCIALIZADORA AGRICOLA S.A.", "COMERCIALIZADORA AGRICOLA"),
        ("COOPERATIVA DE TRANSPORTE S.C. DE R.L.", "COOPERATIVA DE TRANSPORTE"),
        ("BANCO MERCANTIL SAB DE CV", "BANCO MERCANTIL"),
        ("CORPORATIVO NACIONAL, S.A. B. DE C.V.", "CORPORATIVO NACIONAL"),
        ("FARMACIAS DE MÉXICO, S.A. DE C.V.", "FARMACIAS DE MEXICO"),
    ]
    for raw, expected in test_cases:
        result = normalize_mexican_legal_name(raw)
        assert result == expected, f"Failed on '{raw}': got '{result}', expected '{expected}'"


def test_strip_corporate_branch_qualifiers():
    assert strip_corporate_branch_qualifiers("AUMOVIO MEXICO PLANTA NORTE S.A. DE C.V.") == "AUMOVIO"
    assert strip_corporate_branch_qualifiers("AUMOVIO MEXICO MATRIZ") == "AUMOVIO"
    assert strip_corporate_branch_qualifiers("SEGLO OPERACIONES LOGISTICAS NORTE") == "SEGLO OPERACIONES LOGISTICAS"


def test_geohash_assignment():
    df = pl.DataFrame({
        "lat": [21.1221, 21.1230],
        "lon": [-101.6826, -101.6830],
        "norm_name": ["EMPRESA UNO", "EMPRESA DOS"]
    })
    df_gh = assign_geohash_block(df, precision=6)
    assert "geohash" in df_gh.columns
    assert len(df_gh["geohash"][0]) == 6
    # Coordinates ~100m apart should share geohash prefix 5
    assert df_gh["geohash"][0][:5] == df_gh["geohash"][1][:5]


def test_entity_id_generation():
    df = pl.DataFrame({
        "lat": [21.1221],
        "lon": [-101.6826],
        "norm_name": ["TECNOLOGIA DEL CENTRO"],
        "geohash": ["9exyzz"]
    })
    df_ent = assign_entity_ids(df)
    assert "entity_id" in df_ent.columns
    assert df_ent["entity_id"][0].startswith("ent_")


def test_fuzzy_matching_rapidfuzz():
    name_a = "DISTRIBUIDORA DE ACEROS MEXICANA"
    name_b = "DISTRIBUIDORA DE ACEROS DE MEXICO"
    score = fuzz.WRatio(name_a, name_b) / 100.0
    assert score > 0.85


def test_spatial_join_osm_context():
    denue = pl.DataFrame({
        "id": ["1", "2"],
        "lat": [20.9538, 20.9990],
        "lon": [-101.4161, -101.5000],
    })
    osm = pl.DataFrame({
        "osm_id": ["osm_001"],
        "lat": [20.95382],
        "lon": [-101.41608],
        "name": ["Plaza Paseo"],
        "amenity": ["cafe"],
        "office": [""],
        "shop": [""],
    })
    joined = spatial_join_osm_context(denue, osm, max_dist_m=50.0)
    assert "nearest_osm_id" in joined.columns
    assert "osm_verified_storefront" in joined.columns
    assert joined["osm_verified_storefront"][0] is True
    assert joined["osm_verified_storefront"][1] is False
    assert joined["nearest_osm_id"][0] == "osm_001"
    assert joined["nearest_osm_id"][1] is None


def test_classify_b2b_email_role():
    assert classify_b2b_email_role("rh@continental.com") == "decision_maker_role"
    assert classify_b2b_email_role("compras.planta@aumovio.com") == "decision_maker_role"
    assert classify_b2b_email_role("contacto@empresa.com.mx") == "generic_role_inbox"
    assert classify_b2b_email_role("juan.perez@continental.com") == "corporate_direct"
    assert classify_b2b_email_role("negocio@gmail.com") == "free_personal"
    assert classify_b2b_email_role(None) == "none"
