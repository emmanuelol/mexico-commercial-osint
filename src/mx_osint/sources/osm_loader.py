from pathlib import Path
import json
import polars as pl
import requests
from loguru import logger
OVERPASS_ENDPOINTS: list[str] = ['https://lz4.overpass-api.de/api/interpreter', 'https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter']

def query_osm_amenities(lat: float, lon: float, radius_km: float, cache_path: Path=Path('data/processed/osm_amenities.parquet')) -> pl.DataFrame:
    """
    Queries OpenStreetMap Overpass API for commercial/institutional amenities near target location.
    Persisted separately to maintain strict ODbL horizontal separation from INEGI DENUE records.
    """
    if cache_path.exists() and cache_path.stat().st_size > 0:
        df_cached = pl.read_parquet(cache_path)
        logger.info(f'Cache hit for ODbL-separated OSM layer: {cache_path} ({len(df_cached):,} POIs)')
        return df_cached
    radius_m = int(radius_km * 1000)
    query = f'\n    [out:json][timeout:45];\n    (\n      node["amenity"~"^(cafe|restaurant|bank|pharmacy|hospital|clinic|school|university)$"](around:{radius_m},{lat},{lon});\n      way["amenity"~"^(cafe|restaurant|bank|pharmacy|hospital|clinic|school|university)$"](around:{radius_m},{lat},{lon});\n      node["office"](around:{radius_m},{lat},{lon});\n      way["office"](around:{radius_m},{lat},{lon});\n      node["shop"~"^(supermarket|convenience|mall)$"](around:{radius_m},{lat},{lon});\n      way["shop"~"^(supermarket|convenience|mall)$"](around:{radius_m},{lat},{lon});\n    );\n    out center;\n    '
    headers = {'User-Agent': 'MexicoCommercialOSINT/1.0 (ODbL-Layer; Bajio Hub, Gto)', 'Accept': 'application/json'}
    elements = []
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            logger.info(f'Querying OSM Overpass API ({endpoint}) around ({lat:.5f}, {lon:.5f}), radius={radius_km}km...')
            response = requests.post(endpoint, data={'data': query}, headers=headers, timeout=45)
            response.raise_for_status()
            elements = response.json().get('elements', [])
            if elements:
                break
        except Exception as e:
            logger.warning(f'Overpass mirror {endpoint} failed ({e}), trying next...')
    records = []
    for elem in elements:
        lat_val = elem.get('lat') or elem.get('center', {}).get('lat')
        lon_val = elem.get('lon') or elem.get('center', {}).get('lon')
        if lat_val and lon_val:
            tags = elem.get('tags', {})
            records.append({'osm_id': str(elem['id']), 'type': elem['type'], 'lat': float(lat_val), 'lon': float(lon_val), 'name': tags.get('name', ''), 'amenity': tags.get('amenity', ''), 'office': tags.get('office', ''), 'shop': tags.get('shop', '')})
    if not records:
        return pl.DataFrame()
    df_osm = pl.DataFrame(records).unique(subset=['osm_id'], keep='first')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_osm.write_parquet(cache_path)
    logger.info(f'✅ Persisted {len(df_osm):,} unique OSM amenities -> {cache_path}')
    return df_osm


class OSMLoader:
    """
    Standardized loader and Overpass API query interface for OpenStreetMap layers.
    Maintains ODbL horizontal separation and supports Parquet and GeoJSON files.
    """

    def __init__(self, cache_path: Path | str = Path('data/processed/osm_amenities.parquet')):
        self.cache_path = Path(cache_path)

    def load(self, source_path: Path | str) -> pl.DataFrame:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f'OSM source file not found: {path}')
        if path.suffix == '.parquet':
            return pl.read_parquet(path)
        elif path.suffix in ('.geojson', '.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            records = []
            for feat in data.get('features', []):
                props = feat.get('properties', {})
                geom = feat.get('geometry', {})
                coords = geom.get('coordinates', [0.0, 0.0])
                records.append({
                    'osm_id': str(props.get('osm_id', '')),
                    'type': feat.get('type', 'node'),
                    'lat': float(coords[1]),
                    'lon': float(coords[0]),
                    'name': str(props.get('name', '')),
                    'amenity': str(props.get('amenity', '')),
                    'office': str(props.get('office', '')),
                    'shop': str(props.get('shop', '')),
                    'opening_hours': str(props.get('opening_hours', '')),
                })
            if not records:
                return pl.DataFrame()
            return pl.DataFrame(records).unique(subset=['osm_id'], keep='first')
        raise ValueError(f'Unsupported OSM file extension: {path.suffix}')

    def query(self, lat: float, lon: float, radius_km: float) -> pl.DataFrame:
        return query_osm_amenities(lat, lon, radius_km, cache_path=self.cache_path)

