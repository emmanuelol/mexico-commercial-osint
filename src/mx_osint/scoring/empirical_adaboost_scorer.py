from mx_osint.config import Settings
settings = Settings()
from mx_osint.models.denue import STRATA_HEADCOUNT_MIDPOINT

import math
import numpy as np
import polars as pl
import requests
from loguru import logger
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
MORNING_COMMUTE_CORRIDORS: list[dict] = [{'id': 'fed45_norte', 'name': 'Corredor Fed. 45 Norte (León / Puerto Interior ➔ Bajío Hub)', 'tdpa_total': 68500, 'pct_light_cars': 0.72, 'morning_share': 0.24, 'color': '#d62728', 'waypoints': [(20.9785, -101.4485), (20.9685, -101.439), (20.9575, -101.4265), (settings.TARGET_LAT, settings.TARGET_LON)], 'gauge_lat': 20.9685, 'gauge_lon': -101.439}, {'id': 'fed45_sur', 'name': 'Corredor Fed. 45 Sur (Irapuato / FIPASI ➔ Bajío Hub)', 'tdpa_total': 54200, 'pct_light_cars': 0.68, 'morning_share': 0.24, 'color': '#ff7f0e', 'waypoints': [(20.922, -101.4235), (20.9385, -101.4275), (20.9505, -101.4252), (settings.TARGET_LAT, settings.TARGET_LON)], 'gauge_lat': 20.925, 'gauge_lon': -101.4245}, {'id': 'fed110_este', 'name': 'Corredor Fed. 110 / Blvd. GM (Gto. Capital ➔ Bajío Hub / Colinas)', 'tdpa_total': 34800, 'pct_light_cars': 0.81, 'morning_share': 0.24, 'color': '#fdae61', 'waypoints': [(20.956, -101.386), (20.9515, -101.402), (settings.TARGET_LAT, settings.TARGET_LON)], 'gauge_lat': 20.9515, 'gauge_lon': -101.402}, {'id': 'zapata_local', 'name': 'Arteria Local Blvd. Emiliano Zapata (Bajío Centro ➔ Bajío Hub ➔ Parque Ind.)', 'tdpa_total': 28400, 'pct_light_cars': 0.85, 'morning_share': 0.24, 'color': '#00b4d8', 'waypoints': [(20.943, -101.427), (20.9495, -101.421), (settings.TARGET_LAT, settings.TARGET_LON), (20.9682, -101.4243)], 'gauge_lat': 20.9495, 'gauge_lon': -101.421}]
INEGI_IMSS_SCIAN_WAGES_MXN: dict[str, dict[str, float]] = {'3361': {'monthly_wage_mxn': 34500.0, 'deg_share': 0.42}, '3363': {'monthly_wage_mxn': 26800.0, 'deg_share': 0.35}, '3364': {'monthly_wage_mxn': 32000.0, 'deg_share': 0.48}, '3254': {'monthly_wage_mxn': 28500.0, 'deg_share': 0.45}, '5221': {'monthly_wage_mxn': 29400.0, 'deg_share': 0.78}, '5413': {'monthly_wage_mxn': 27200.0, 'deg_share': 0.82}, '5415': {'monthly_wage_mxn': 33100.0, 'deg_share': 0.88}, '5416': {'monthly_wage_mxn': 26500.0, 'deg_share': 0.8}, '6221': {'monthly_wage_mxn': 24800.0, 'deg_share': 0.7}, '7211': {'monthly_wage_mxn': 16500.0, 'deg_share': 0.28}, '55': {'monthly_wage_mxn': 36000.0, 'deg_share': 0.85}, '52': {'monthly_wage_mxn': 25200.0, 'deg_share': 0.72}, '54': {'monthly_wage_mxn': 23800.0, 'deg_share': 0.75}, '51': {'monthly_wage_mxn': 22400.0, 'deg_share': 0.65}, '62': {'monthly_wage_mxn': 19500.0, 'deg_share': 0.6}, '33': {'monthly_wage_mxn': 18900.0, 'deg_share': 0.26}, '32': {'monthly_wage_mxn': 17200.0, 'deg_share': 0.22}, '56': {'monthly_wage_mxn': 15800.0, 'deg_share': 0.3}, '43': {'monthly_wage_mxn': 16400.0, 'deg_share': 0.28}, '48': {'monthly_wage_mxn': 16900.0, 'deg_share': 0.2}, '61': {'monthly_wage_mxn': 15200.0, 'deg_share': 0.85}, '93': {'monthly_wage_mxn': 14800.0, 'deg_share': 0.65}, '31': {'monthly_wage_mxn': 12400.0, 'deg_share': 0.14}, '46': {'monthly_wage_mxn': 10800.0, 'deg_share': 0.15}}

def fetch_osrm_corridor_geometry(waypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Snaps corridor waypoints to physical road geometry via OSRM (falls back to waypoints if offline)."""
    coords_str = ';'.join((f'{lon},{lat}' for lat, lon in waypoints))
    url = f'https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson'
    try:
        resp = requests.get(url, headers={'User-Agent': 'MexicoCommercialOSINT/1.0'}, timeout=6)
        resp.raise_for_status()
        raw_coords = resp.json()['routes'][0]['geometry']['coordinates']
        return [(float(lat), float(lon)) for lon, lat in raw_coords]
    except Exception as e:
        logger.warning(f'OSRM corridor fallback to waypoints ({e})')
        return waypoints

def build_corridor_geometries() -> tuple[list[dict], list[tuple[float, float, float]]]:
    """Builds road-snapped SICT 7:00-10:00 AM corridors and sampled intensity nodes."""
    enriched = []
    traffic_nodes: list[tuple[float, float, float]] = []
    max_morning_cars = max((c['tdpa_total'] * c['pct_light_cars'] * c['morning_share'] for c in MORNING_COMMUTE_CORRIDORS))
    for corr in MORNING_COMMUTE_CORRIDORS:
        morning_cars = int(corr['tdpa_total'] * corr['pct_light_cars'] * corr['morning_share'])
        norm_intensity = morning_cars / max_morning_cars
        geometry = fetch_osrm_corridor_geometry(corr['waypoints'])
        enriched.append({**corr, 'morning_cars': morning_cars, 'norm_intensity': norm_intensity, 'geometry': geometry})
        step = max(1, len(geometry) // 25)
        for lat, lon in geometry[::step]:
            traffic_nodes.append((lat, lon, norm_intensity))
    return (enriched, traffic_nodes)

def enrich_empirical_income_and_scholarship(df: pl.DataFrame, traffic_nodes: list[tuple[float, float, float]]) -> pl.DataFrame:
    """
    Enriches DENUE records with empirical INEGI/IMSS wage, university degree share,
    estimated monthly payroll pool, and 7:00-10:00 AM SICT commuter exposure.
    """
    wages, deg_shares, payrolls, traffic_exp, corridor_dists = ([], [], [], [], [])
    r_km, rad = (6371.0, math.pi / 180.0)
    for row in df.iter_rows(named=True):
        scian6 = str(row['scian_code']).zfill(6)
        scian4, scian2 = (scian6[:4], scian6[:2])
        emp_data = INEGI_IMSS_SCIAN_WAGES_MXN.get(scian4) or INEGI_IMSS_SCIAN_WAGES_MXN.get(scian2) or {'monthly_wage_mxn': 13500.0, 'deg_share': 0.2}
        wage = emp_data['monthly_wage_mxn']
        deg = emp_data['deg_share']
        email_str = (row.get('email') or '').lower()
        has_corp_domain = '@' in email_str and (not any((free in email_str for free in ['gmail.', 'hotmail.', 'outlook.', 'yahoo.', 'live.', 'gob.mx', 'prodigy.'])))
        if has_corp_domain:
            wage *= 1.12
            deg = min(0.95, deg + 0.1)
        headcount = STRATA_HEADCOUNT_MIDPOINT.get(row['employee_strata'], 15.0)
        wages.append(round(wage, 0))
        deg_shares.append(round(deg, 2))
        payrolls.append(round(wage * headcount, 0))
        b_lat, b_lon = (row['lat'], row['lon'])
        best_corr = 0.0
        min_corr_dist = 99.0
        for n_lat, n_lon, norm_int in traffic_nodes:
            dlat, dlon = ((n_lat - b_lat) * rad, (n_lon - b_lon) * rad)
            a = math.sin(dlat / 2) ** 2 + math.cos(b_lat * rad) * math.cos(n_lat * rad) * math.sin(dlon / 2) ** 2
            d_km = 2 * math.asin(math.sqrt(a)) * r_km
            if d_km < min_corr_dist:
                min_corr_dist = d_km
            best_corr = max(best_corr, norm_int * math.exp(-d_km / 0.8))
        paseo_prox = max(0.15, 1.0 - row['distance_km'] / settings.RADIUS_KM)
        traffic_exp.append(round((0.65 * best_corr + 0.35 * paseo_prox) * 10.0, 2))
        corridor_dists.append(round(min_corr_dist, 2))
    return df.with_columns([pl.Series('est_monthly_wage_mxn', wages), pl.Series('est_degree_share', deg_shares), pl.Series('est_monthly_payroll_mxn', payrolls), pl.Series('morning_traffic_score', traffic_exp), pl.Series('corridor_dist_km', corridor_dists)])

def calculate_critic_and_adaboost_weights(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
    """
    Computes CRITIC Objective Weights (unsupervised variance x conflict) and
    AdaBoost Classifier Weights (supervised seed archetypes), producing `b2b_5d_score`.
    """
    status_map = {'target': 1.0, 'partial': 0.5, 'no_target': 0.1}
    df_feat = df.with_columns([pl.when((pl.col('b2b_status') == 'partial') & (pl.col('est_monthly_wage_mxn') >= 24000) & (pl.col('est_degree_share') >= 0.35)).then(pl.lit(0.85)).otherwise(pl.col('b2b_status').replace_strict(status_map, default=0.1)).alias('f_b2b_fit'), pl.col('est_monthly_wage_mxn').alias('f_wage_income'), (pl.col('est_degree_share') * pl.col('est_monthly_payroll_mxn')).log1p().alias('f_white_collar_mass'), pl.col('morning_traffic_score').alias('f_traffic_7_10am'), pl.when(pl.col('email').is_not_null() & (pl.col('email') != '')).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias('f_contact_ready')])
    feature_cols = ['f_b2b_fit', 'f_wage_income', 'f_white_collar_mass', 'f_traffic_7_10am', 'f_contact_ready']
    raw_matrix = df_feat.select(feature_cols).to_numpy()
    scaler = MinMaxScaler()
    norm_matrix = scaler.fit_transform(raw_matrix)
    std_devs = np.std(norm_matrix, axis=0, ddof=1)
    corr_matrix = np.nan_to_num(np.corrcoef(norm_matrix, rowvar=False), nan=0.0)
    conflict = np.sum(1.0 - corr_matrix, axis=1)
    critic_info = std_devs * conflict
    critic_weights = critic_info / np.sum(critic_info)
    y_seed = ((df_feat['f_b2b_fit'] >= 0.85) & (df_feat['est_monthly_wage_mxn'] >= 22000) & (df_feat['distance_km'] <= 3.0)).to_numpy().astype(int)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.5, random_state=42)
    ada.fit(norm_matrix, y_seed)
    ada_weights = ada.feature_importances_
    ada_probs = ada.predict_proba(norm_matrix)[:, 1]
    critic_scores = norm_matrix @ critic_weights * 10.0
    ada_scores = 0.5 * (norm_matrix @ ada_weights * 10.0) + 0.5 * (ada_probs * 10.0)
    weight_comparison = {col: {'CRITIC_Objective': round(float(cw), 3), 'AdaBoost_Learned': round(float(aw), 3)} for col, cw, aw in zip(feature_cols, critic_weights, ada_weights)}
    df_out = df_feat.with_columns([pl.Series('critic_objective_score', np.round(critic_scores, 2)), pl.Series('adaboost_b2b_score', np.round(ada_scores, 2)), pl.Series('b2b_5d_score', np.round((critic_scores + ada_scores) / 2.0, 2)), (pl.col('f_b2b_fit') == 0.85).alias('is_rescued_partial'), (pl.lit('$') + (pl.col('est_monthly_wage_mxn') / 1000).round(1).cast(pl.Utf8) + pl.lit('k MXN/mes (') + (pl.col('est_degree_share') * 100).round(0).cast(pl.Int32).cast(pl.Utf8) + pl.lit('% Prof.)')).alias('workforce_scholarship_tier')]).drop(feature_cols).sort('b2b_5d_score', descending=True)
    rescued_cnt = df_out.filter(pl.col('is_rescued_partial')).height
    logger.info(f'✅ Empirical 5D Scoring (CRITIC + AdaBoost) complete: {rescued_cnt} high-wage Partial Targets rescued.')
    return (df_out, weight_comparison)


class EmpiricalAdaBoostScorer:
    """
    Empirical 5D Multi-Criteria Prospect Scorer fusing unsupervised CRITIC objective weighting
    with supervised AdaBoost decision tree classifiers to calibrate B2B conversion scores.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def score(
        self,
        df: pl.DataFrame,
        traffic_nodes: list[tuple[float, float, float]] | None = None,
    ) -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
        """Calculates CRITIC and AdaBoost 5D scores for candidate business entities."""
        if traffic_nodes is None:
            _, traffic_nodes = build_corridor_geometries()
        if 'est_monthly_wage_mxn' not in df.columns:
            df = enrich_empirical_income_and_scholarship(df, traffic_nodes)
        if 'b2b_status' not in df.columns:
            from mx_osint.scoring.b2b_qualifier import qualify_b2b_target
            df = df.with_columns(
                pl.struct(['scian_code', 'employee_strata'])
                .map_elements(
                    lambda s: qualify_b2b_target(s.get('scian_code'), s.get('employee_strata')),
                    return_dtype=pl.Utf8,
                )
                .alias('b2b_status')
            )
        return calculate_critic_and_adaboost_weights(df)

