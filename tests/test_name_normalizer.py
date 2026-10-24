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
