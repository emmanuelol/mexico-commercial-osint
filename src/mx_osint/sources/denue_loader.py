import hashlib
import io
import math
import shutil
import zipfile
from pathlib import Path
import polars as pl
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
DENUE_COLUMN_MAPPING: dict[str, str] = {'id': 'row_id', 'clee': 'denue_id', 'nom_estab': 'business_name', 'raz_social': 'legal_name', 'codigo_act': 'scian_code', 'nombre_act': 'scian_description', 'per_ocu': 'employee_strata', 'tipo_vial': 'street_type', 'nom_vial': 'street_name', 'numero_ext': 'street_number_ext', 'cod_postal': 'postal_code', 'entidad': 'state', 'municipio': 'municipality', 'localidad': 'locality', 'telefono': 'phone', 'correoelec': 'email', 'www': 'website', 'fecha_alta': 'registration_date', 'latitud': 'lat', 'longitud': 'lon', 'cve_ent': 'cve_ent', 'cve_mun': 'cve_mun'}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_denue_state(state_code: str, cache_dir: Path) -> Path:
    """Downloads DENUE state zip package from official INEGI portal with retry and caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_file = cache_dir / f'denue_{state_code}_csv.zip'
    alt_cache = Path('../data/raw') / f'denue_{state_code}_csv.zip'
    nb_cache = Path('notebooks/data/raw') / f'denue_{state_code}_csv.zip'
    for candidate in (target_file, alt_cache, nb_cache):
        if candidate.exists() and candidate.stat().st_size > 0:
            logger.info(f'Cache hit for state {state_code}: {candidate}')
            return candidate
    url = f'https://www.inegi.org.mx/contenidos/masiva/denue/denue_{state_code}_csv.zip'
    logger.info(f'Downloading DENUE state {state_code} from {url}...')
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    hasher = hashlib.sha256()
    with open(target_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                hasher.update(chunk)
    logger.info(f'Saved {target_file} (SHA256: {hasher.hexdigest()[:12]})')
    return target_file

def read_denue_csv(zip_path: Path) -> pl.DataFrame:
    """Reads INEGI DENUE zip CSV with latin1 decoding, type normalization, column pruning, and CLEE deduplication."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        data_files = [f for f in zf.namelist() if f.startswith('conjunto_de_datos/') and f.endswith('.csv')]
        if not data_files:
            raise ValueError(f'No data CSV found in {zip_path}.')
        data_file = data_files[0]
        logger.info(f'Reading DENUE data file: {data_file}')
        with zf.open(data_file) as f:
            raw_bytes = f.read()
    df = pl.read_csv(io.BytesIO(raw_bytes), encoding='latin1', separator=',', quote_char='"', has_header=True, infer_schema_length=10000, ignore_errors=True, null_values=['', 'NA', 'N/A', 'NULL'])
    rename_dict = {old: new for old, new in DENUE_COLUMN_MAPPING.items() if old in df.columns}
    missing = {'lat', 'lon', 'business_name'} - set(rename_dict.values())
    if missing:
        raise ValueError(f'Critical columns missing: {missing}')
    df = df.rename(rename_dict)
    keep_cols = [c for c in DENUE_COLUMN_MAPPING.values() if c in df.columns]
    df = df.select(keep_cols)
    str_cols = [c for c in ['row_id', 'denue_id', 'business_name', 'legal_name', 'scian_description', 'employee_strata', 'email', 'website', 'phone', 'street_type', 'street_name', 'street_number_ext', 'postal_code', 'state', 'municipality', 'locality'] if c in df.columns]
    raw_count = len(df)
    df = df.with_columns([pl.col('scian_code').cast(pl.Utf8, strict=False).str.strip_chars().str.zfill(6), pl.col('lat').cast(pl.Utf8, strict=False).str.replace(',', '.').str.strip_chars().cast(pl.Float64, strict=False), pl.col('lon').cast(pl.Utf8, strict=False).str.replace(',', '.').str.strip_chars().cast(pl.Float64, strict=False), *[pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars() for c in str_cols]]).filter(pl.col('lat').is_not_null() & pl.col('lon').is_not_null() & pl.col('lat').is_between(-90, 90) & pl.col('lon').is_between(-180, 180) & pl.col('business_name').is_not_null() & (pl.col('business_name') != '')).unique(subset=['denue_id'], keep='first', maintain_order=True)
    logger.info(f'✅ Loaded & deduplicated DENUE DataFrame: {len(df):,} unique valid rows (removed {raw_count - len(df):,} invalid/duplicate rows) × {len(df.columns)} standardized columns')
    return df

def filter_by_radius(df: pl.DataFrame, target_lat: float, target_lon: float, radius_km: float) -> pl.DataFrame:
    """Vectorized Haversine distance calculation and radius filtering using Polars expression API."""
    r_km = 6371.0
    rad = math.pi / 180.0
    t_lat_rad = target_lat * rad
    lat_col = pl.col('lat').cast(pl.Float64, strict=False)
    lon_col = pl.col('lon').cast(pl.Float64, strict=False)
    dlat = (lat_col - target_lat) * rad
    dlon = (lon_col - target_lon) * rad
    a = (dlat / 2.0).sin().pow(2) + math.cos(t_lat_rad) * (lat_col * rad).cos() * (dlon / 2.0).sin().pow(2)
    dist_expr = (2.0 * a.sqrt().arcsin() * r_km).alias('distance_km')
    return df.with_columns(dist_expr).filter(pl.col('distance_km') <= radius_km).sort('distance_km')

def filter_by_strata(df: pl.DataFrame, min_strata: str) -> pl.DataFrame:
    """Filters records by employee strata threshold or higher."""
    if min_strata not in STRATA_ORDER:
        raise ValueError(f"Invalid strata '{min_strata}'. Must be one of {STRATA_ORDER}")
    allowed = STRATA_ORDER[STRATA_ORDER.index(min_strata):]
    return df.filter(pl.col('employee_strata').is_in(allowed))

def persist_stage1_extract(df: pl.DataFrame, state_code: str, radius_km: float, raw_dir: Path, processed_dir: Path) -> tuple[Path, Path]:
    """
    Persists Stage-01 enriched DENUE extract to processed_dir and writes the
    BusinessEntity-compatible handoff parquet to raw_dir for Stage 02 (02_entity_resolution.ipynb).
    """
    filename = f'denue_state_{state_code}_radius_{int(radius_km)}km.parquet'
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_path = processed_dir / filename
    df.write_parquet(proc_path)
    handoff_df = df.with_columns([pl.col('row_id').alias('id'), pl.col('denue_id').alias('clee'), pl.col('business_name').alias('nombre_estab'), pl.col('legal_name').alias('raz_social'), pl.col('scian_code').alias('codigo_act'), pl.col('scian_description').alias('nombre_act'), pl.col('employee_strata').alias('per_ocu'), pl.col('phone').alias('telefono'), pl.col('email').alias('correoelec'), pl.col('website').alias('www'), (pl.col('cve_ent').cast(pl.Utf8).str.zfill(2) if 'cve_ent' in df.columns else pl.lit(state_code)).alias('ent'), (pl.col('cve_mun').cast(pl.Utf8).str.zfill(3) if 'cve_mun' in df.columns else pl.col('municipality')).alias('mun'), pl.col('registration_date').alias('fecha_alta')])
    raw_path = raw_dir / filename
    handoff_df.write_parquet(raw_path)
    for peer_root in (Path('..'), Path('notebooks')):
        if (peer_root / 'data').exists():
            peer_raw = peer_root / 'data' / 'raw' / filename
            peer_proc = peer_root / 'data' / 'processed' / filename
            peer_raw.parent.mkdir(parents=True, exist_ok=True)
            peer_proc.parent.mkdir(parents=True, exist_ok=True)
            if peer_raw.resolve() != raw_path.resolve():
                shutil.copy2(raw_path, peer_raw)
            if peer_proc.resolve() != proc_path.resolve():
                shutil.copy2(proc_path, peer_proc)
    logger.info(f'✅ Persisted Stage-01 processed extract ({len(df):,} rows) -> {proc_path}')
    logger.info(f'✅ Persisted Stage-02 handoff extract ({len(handoff_df):,} rows) -> {raw_path}')
    return (proc_path, raw_path)
