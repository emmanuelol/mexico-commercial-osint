import math
from pathlib import Path
import polars as pl
from IPython.display import HTML, display

def spatial_join_osm_context(denue_df: pl.DataFrame, osm_df: pl.DataFrame, max_dist_m: float=50.0) -> pl.DataFrame:
    """Cross-source spatial resolution between DENUE and OSM amenities respecting ODbL boundaries."""
    if len(osm_df) == 0:
        return denue_df.with_columns([pl.lit(None).cast(pl.Utf8).alias('nearest_osm_id'), pl.lit(None).cast(pl.Float64).alias('osm_match_dist_m'), pl.lit(False).alias('osm_verified_storefront'), pl.lit('').alias('osm_poi_context')])
    osm_records = osm_df.select(['osm_id', 'lat', 'lon', 'name', 'amenity', 'office', 'shop']).to_dicts()
    r_earth_m = 6371000.0
    rad = math.pi / 180.0
    nearest_ids = []
    nearest_dists = []
    verified_flags = []
    contexts = []
    for row in denue_df.iter_rows(named=True):
        d_lat, d_lon = (row['lat'], row['lon'])
        best_dist = float('inf')
        best_osm = None
        for osm in osm_records:
            o_lat, o_lon = (osm['lat'], osm['lon'])
            dlat = (o_lat - d_lat) * rad
            dlon = (o_lon - d_lon) * rad
            a = math.sin(dlat / 2.0) ** 2 + math.cos(d_lat * rad) * math.cos(o_lat * rad) * math.sin(dlon / 2.0) ** 2
            dist_m = 2.0 * math.asin(math.sqrt(a)) * r_earth_m
            if dist_m < best_dist:
                best_dist = dist_m
                best_osm = osm
        if best_osm and best_dist <= 250.0:
            nearest_ids.append(str(best_osm['osm_id']))
            nearest_dists.append(round(best_dist, 1))
            verified_flags.append(best_dist <= max_dist_m)
            ctx = best_osm.get('name') or best_osm.get('amenity') or best_osm.get('office') or best_osm.get('shop') or ''
            contexts.append(str(ctx))
        else:
            nearest_ids.append(None)
            nearest_dists.append(None)
            verified_flags.append(False)
            contexts.append('')
    return denue_df.with_columns([pl.Series('nearest_osm_id', nearest_ids, dtype=pl.Utf8), pl.Series('osm_match_dist_m', nearest_dists, dtype=pl.Float64), pl.Series('osm_verified_storefront', verified_flags, dtype=pl.Boolean), pl.Series('osm_poi_context', contexts, dtype=pl.Utf8)])


class SpatialBlocker:
    """
    Haversine spatial blocker for candidate pair reduction and cross-source verification
    between INEGI DENUE establishment coordinates and OpenStreetMap POIs.
    """

    def __init__(self, max_dist_m: float = 50.0):
        self.max_dist_m = max_dist_m

    def block(self, denue_df: pl.DataFrame, osm_df: pl.DataFrame) -> pl.DataFrame:
        """Executes spatial candidate bounding and storefront context attachment."""
        return spatial_join_osm_context(denue_df, osm_df, max_dist_m=self.max_dist_m)

