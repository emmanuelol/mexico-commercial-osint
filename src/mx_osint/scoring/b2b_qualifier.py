import polars as pl
from loguru import logger
SCIAN_SECTOR_MAP: dict[str, str] = {'51': 'information_media', '52': 'financial_services', '53': 'real_estate', '54': 'professional_services', '55': 'corporate_offices', '56': 'business_support', '62': 'healthcare', '23': 'construction', '31': 'manufacturing', '32': 'manufacturing', '33': 'manufacturing', '43': 'wholesale_trade', '48': 'transportation', '49': 'transportation', '61': 'education', '72': 'hospitality', '11': 'agriculture', '21': 'mining', '22': 'utilities', '46': 'retail_trade', '71': 'entertainment', '81': 'other_services', '93': 'government'}
TARGET_SECTORS: set[str] = {'professional_services', 'corporate_offices', 'financial_services', 'information_media', 'real_estate', 'business_support', 'healthcare'}
PARTIAL_SECTORS: set[str] = {'manufacturing', 'hospitality', 'education', 'construction', 'wholesale_trade', 'transportation'}

def qualify_b2b(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classifies DENUE records into 'target', 'partial', or 'no_target' using 2-digit SCIAN prefixes
    and excludes public-sector institutions and food/beverage retailers from private B2B outreach.
    """
    sector_lookup = pl.DataFrame({'scian_sector_code': list(SCIAN_SECTOR_MAP.keys()), 'b2b_sector': list(SCIAN_SECTOR_MAP.values())})
    is_public_entity = pl.col('scian_description').fill_null('').str.to_lowercase().str.contains('sector público|asistencia social') | pl.col('email').fill_null('').str.to_lowercase().str.contains('\\.gob\\.mx') | pl.col('website').fill_null('').str.to_lowercase().str.contains('\\.gob\\.mx') | pl.col('b2b_sector').is_in(['education', 'healthcare']) & pl.col('business_name').fill_null('').str.to_lowercase().str.contains('públic|public|oficial|federal|estatal|municipal|imss|issste|isapeg|dif |secretar')
    df_enriched = df.with_columns(pl.col('scian_code').cast(pl.Utf8).str.zfill(6).str.slice(0, 2).alias('scian_sector_code')).join(sector_lookup, on='scian_sector_code', how='left').with_columns(pl.col('b2b_sector').fill_null('unclassified')).with_columns(pl.when(pl.col('b2b_sector').is_in(TARGET_SECTORS)).then(pl.lit('target')).when(pl.col('b2b_sector').is_in(PARTIAL_SECTORS)).then(pl.lit('partial')).otherwise(pl.lit('no_target')).alias('b2b_status')).with_columns(pl.when(is_public_entity).then(pl.lit('no_target')).when((pl.col('b2b_sector') == 'hospitality') & ~pl.col('scian_code').str.starts_with('721')).then(pl.lit('no_target')).otherwise(pl.col('b2b_status')).alias('b2b_status'))
    logger.info('✅ SCIAN B2B qualification complete (public sector & food/bev retail excluded).')
    return df_enriched
