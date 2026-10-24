from pathlib import Path
import math
import polars as pl
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

STRATA_ORDER = [
    "0 a 5 personas",
    "6 a 10 personas",
    "11 a 30 personas",
    "31 a 50 personas",
    "51 a 100 personas",
    "101 a 250 personas",
    "251 y más personas"
]

def haversine_distance_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def filter_by_radius(df: pl.DataFrame, target_lat: float, target_lon: float, radius_km: float) -> pl.DataFrame:
    # Vectorized Haversine in Polars expression API
    r_km = 6371.0
    rad = math.pi / 180.0
    
    t_lat_rad = target_lat * rad
    t_lon_rad = target_lon * rad
    
    lat_rad = pl.col("lat") * rad
    lon_rad = pl.col("lon") * rad
    
    dlat = (pl.col("lat") - target_lat) * rad
    dlon = (pl.col("lon") - target_lon) * rad
    
    a = (
        (dlat / 2.0).sin().pow(2)
        + math.cos(t_lat_rad) * lat_rad.cos() * (dlon / 2.0).sin().pow(2)
    )
    c = 2.0 * a.sqrt().arcsin()
    dist_expr = (c * r_km).alias("distance_km")
    
    return (
        df.with_columns(dist_expr)
        .filter(pl.col("distance_km") <= radius_km)
        .sort("distance_km")
    )

def filter_by_strata(df: pl.DataFrame, min_strata: str) -> pl.DataFrame:
    if min_strata not in STRATA_ORDER:
        raise ValueError(f"Unknown strata {min_strata}")
    allowed = STRATA_ORDER[STRATA_ORDER.index(min_strata):]
    return df.filter(pl.col("per_ocu").is_in(allowed))


def test_haversine_known_coordinates():
    # Distance to self is 0
    d_zero = haversine_distance_scalar(21.1221, -101.6826, 21.1221, -101.6826)
    assert round(d_zero, 4) == 0.0

    # Approx 1 deg latitude ≈ 111.19 km
    d_deg = haversine_distance_scalar(21.1221, -101.6826, 22.1221, -101.6826)
    assert 110.0 < d_deg < 112.0


def test_filter_by_radius_vectorized():
    fixture_path = ROOT_DIR / "fixtures" / "synthetic_denue_sample.parquet"
    df = pl.read_parquet(fixture_path)
    
    filtered = filter_by_radius(df, 21.1221, -101.6826, radius_km=2.0)
    assert len(filtered) > 0
    assert len(filtered) <= len(df)
    assert "distance_km" in filtered.columns
    # Distances are all <= 2.0 and sorted ascending
    dists = filtered["distance_km"].to_list()
    assert all(d <= 2.0 for d in dists)
    assert dists == sorted(dists)


def test_filter_by_strata():
    fixture_path = ROOT_DIR / "fixtures" / "synthetic_denue_sample.parquet"
    df = pl.read_parquet(fixture_path)
    
    filtered = filter_by_strata(df, min_strata="31 a 50 personas")
    allowed = {"31 a 50 personas", "51 a 100 personas", "101 a 250 personas", "251 y más personas"}
    present_strata = set(filtered["per_ocu"].unique().to_list())
    assert present_strata.issubset(allowed)
    assert "0 a 5 personas" not in present_strata
    assert "6 a 10 personas" not in present_strata
    assert "11 a 30 personas" not in present_strata
