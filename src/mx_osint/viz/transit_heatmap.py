from mx_osint.config import Settings
settings = Settings()

from pathlib import Path
import folium
from folium.plugins import AntPath, HeatMap
from loguru import logger
import polars as pl
import requests

def fetch_osrm_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> list[tuple[float, float]]:
    """Fetches driving route geometry via OSRM between Anchor Hub and a prospect."""
    url = f'https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson'
    try:
        resp = requests.get(url, headers={'User-Agent': 'MexicoCommercialOSINT/1.0'}, timeout=6)
        resp.raise_for_status()
        coords = resp.json()['routes'][0]['geometry']['coordinates']
        return [(float(lat), float(lon)) for lon, lat in coords]
    except Exception:
        return []

def create_b2b_osint_map(df_scored: pl.DataFrame, corridors: list[dict], traffic_nodes: list[tuple[float, float, float]], osm_df: pl.DataFrame | None=None, target_lat: float=settings.TARGET_LAT, target_lon: float=settings.TARGET_LON, output_path: Path=settings.PROCESSED_DIR / 'b2b_5d_morning_transit_heatmap.html') -> folium.Map:
    """
    Builds the unified Interactive OSINT B2B Map combining:
      1. Open Base Layers (CartoDB Positron/Dark Matter direct tiles, Esri, Google Traffic)
      2. 1 km and 3 km Catchment Rings
      3. 5D Empirical HeatMap (CRITIC + AdaBoost Ensemble Score + 7:00-10:00 AM SICT Flow)
      4. Animated 7:00-10:00 AM SICT Commute Corridors (AntPath + TDPA Gauges)
      5. OSRM Delivery Routes to Top Actionable B2B Prospects
      6. ODbL-Isolated OpenStreetMap Amenities Layer
      7. Scored B2B Prospect Markers (Direct, Rescued High-Wage, Partial, Excluded)
    """
    m = folium.Map(location=[target_lat, target_lon], zoom_start=14, tiles=None)
    folium.TileLayer(tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', attr='&copy; OpenStreetMap contributors &copy; CARTO', name='🗺️ Mapa Claro Ejecutivo (CARTO Positron)', overlay=False, control=True).add_to(m)
    folium.TileLayer(tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', attr='&copy; OpenStreetMap contributors &copy; CARTO', name='🌙 Modo Oscuro (Alto Contraste Calor)', overlay=False, control=True).add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', attr='Tiles &copy; Esri', name='🏙️ Esri World Street Map', overlay=False, control=True).add_to(m)
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=m,traffic&x={x}&y={y}&z={z}', attr='Google Maps', name='🚦 Google Maps (Tráfico en Tiempo Real)', max_zoom=20, overlay=False, control=True).add_to(m)
    fg_rings = folium.FeatureGroup(name='⭕ Anillos de Cobertura (1 km y 3 km)', show=True)
    for radius_m, label, ring_color in [(1000, 'Radio 1 km (Intercepción Directa / Caminable)', '#d62728'), (3000, 'Radio 3 km (Corredor Urbano / Industrial)', '#ff7f0e')]:
        folium.Circle(location=[target_lat, target_lon], radius=radius_m, color=ring_color, weight=1.8, fill=False, dash_array='6, 6', tooltip=label).add_to(fg_rings)
    fg_rings.add_to(m)
    heat_points: list[list[float]] = []
    for row in df_scored.iter_rows(named=True):
        weight = float(row['b2b_5d_score']) ** 1.6
        if row['b2b_status'] == 'no_target':
            weight *= 0.2
        heat_points.append([row['lat'], row['lon'], weight])
    for t_lat, t_lon, norm_int in traffic_nodes:
        heat_points.append([t_lat, t_lon, norm_int * 8.0])
    HeatMap(heat_points, name='🔥 Mapa de Calor 5D (Salario INEGI/IMSS + Score Ensemble + Flujo 7-10 AM)', min_opacity=0.32, max_zoom=16, radius=28, blur=20, gradient={0.25: '#2b83ba', 0.5: '#abdda4', 0.72: '#fdae61', 0.9: '#f46d43', 1.0: '#d7191c'}, show=True).add_to(m)
    fg_flows = folium.FeatureGroup(name='🚙 Corredores Flujo Matutino 7:00–10:00 AM (SICT)', show=True)
    for corr in corridors:
        AntPath(locations=corr['geometry'], color=corr['color'], pulse_color='#ffffff', weight=5, opacity=0.8, dash_array=[12, 24], delay=550, tooltip=f"<b>🛣️ {corr['name']}</b><br>Flujo 7:00–10:00 AM: <b>{corr['morning_cars']:,} autos ligeros</b> (TDPA Total: {corr['tdpa_total']:,})").add_to(fg_flows)
        folium.CircleMarker(location=[corr['gauge_lat'], corr['gauge_lon']], radius=10, color='#003399', weight=2, fill=True, fill_color='#3385ff', fill_opacity=0.85, tooltip=f"📊 Aforo SICT 7-10 AM: {corr['morning_cars']:,} autos ({corr['name'][:32]}...)", popup=folium.Popup(f"<div style='font-family:Arial; font-size:12px; min-width:210px;'><b style='color:#003399;'>Estación Aforo SICT</b><br>{corr['name']}<hr style='margin:4px 0;'><b>TDPA Diario:</b> {corr['tdpa_total']:,} veh<br><b>Autos Ligeros:</b> {corr['pct_light_cars'] * 100:.0f}%<br><b style='color:#d62728;'>Pico 7:00–10:00 AM:</b> {corr['morning_cars']:,} autos</div>", max_width=300)).add_to(fg_flows)
    fg_flows.add_to(m)
    fg_routes = folium.FeatureGroup(name='📦 Rutas OSRM a Top 8 Cuentas B2B', show=False)
    top_actionable = df_scored.filter((pl.col('b2b_status') == 'target') | pl.col('is_rescued_partial')).head(8)
    for row in top_actionable.iter_rows(named=True):
        route_coords = fetch_osrm_route(target_lat, target_lon, row['lat'], row['lon'])
        if route_coords:
            folium.PolyLine(route_coords, color='#0052cc', weight=3.5, opacity=0.8, tooltip=f"Ruta OSRM a {row['business_name']} ({row['distance_km']:.2f} km)").add_to(fg_routes)
    fg_routes.add_to(m)
    if osm_df is not None and len(osm_df) > 0:
        fg_osm = folium.FeatureGroup(name=f'🌐 Amenidades OSM ODbL ({len(osm_df)})', show=False)
        for r in osm_df.iter_rows(named=True):
            kind = r.get('amenity') or r.get('office') or r.get('shop') or 'POI'
            name_label = r.get('name') or f'OSM {kind}'
            folium.CircleMarker(location=[r['lat'], r['lon']], radius=4.5, color='#008080', fill=True, fill_opacity=0.6, tooltip=f'[OSM {kind}] {name_label}').add_to(fg_osm)
        fg_osm.add_to(m)
    n_target = df_scored.filter(pl.col('b2b_status') == 'target').height
    n_rescued = df_scored.filter(pl.col('is_rescued_partial')).height
    fg_target = folium.FeatureGroup(name=f'✅ Target Directo B2B ({n_target})', show=True)
    fg_rescued = folium.FeatureGroup(name=f'🚀 Rescatado: Alto Salario INEGI/IMSS ({n_rescued})', show=True)
    fg_partial = folium.FeatureGroup(name='⚠️ Target Parcial Estándar', show=False)
    fg_excluded = folium.FeatureGroup(name='❌ No Target', show=False)
    for row in df_scored.iter_rows(named=True):
        status = row['b2b_status']
        is_rescued = row['is_rescued_partial']
        if status == 'target':
            color, radius, fill_op, group = ('#1b7837', 8.5, 0.92, fg_target)
            badge = '✅ TARGET DIRECTO'
        elif is_rescued:
            color, radius, fill_op, group = ('#7b3294', 8.5, 0.92, fg_rescued)
            badge = '🚀 RESCATADO (ALTO SALARIO / ING.)'
        elif status == 'partial':
            color, radius, fill_op, group = ('#e66101', 6.0, 0.75, fg_partial)
            badge = '⚠️ TARGET PARCIAL'
        else:
            color, radius, fill_op, group = ('#878787', 4.0, 0.4, fg_excluded)
            badge = '❌ NO TARGET'
        payroll_m = row['est_monthly_payroll_mxn'] / 1000000.0
        popup_html = f"""\n        <div style="min-width: 275px; font-family: Arial, sans-serif; font-size: 12px;">\n            <b style="color:{color}; font-size:14px;">{row['business_name']}</b><br>\n            <span style="background-color:{color}; color:white; padding:2px 6px; border-radius:3px; font-size:10px; font-weight:bold;">\n                {badge} — Ensemble: {row['b2b_5d_score']:.2f} / 10\n            </span><br>\n            <small style="color:#555;">CRITIC: {row['critic_objective_score']:.2f} | AdaBoost: {row['adaboost_b2b_score']:.2f}</small>\n            <hr style="margin: 6px 0;">\n            <b>💰 Salario Empírico (INEGI/IMSS):</b> ${row['est_monthly_wage_mxn']:,.0f} MXN/mes ({row['est_degree_share'] * 100:.0f}% Prof.)<br>\n            <b>🏦 Masa Salarial Est.:</b> ${payroll_m:.2f}M MXN/mes ({row['employee_strata']})<br>\n            <b>🏢 Sector:</b> {row['b2b_sector']} ({str(row['scian_description'])[:42]}...)<br>\n            <b>🚙 Flujo 7-10 AM:</b> {row['morning_traffic_score']:.1f}/10 (a {row['corridor_dist_km']:.2f} km de corredor)<br>\n            <b>📍 Distancia a Anchor Hub:</b> {row['distance_km']:.2f} km\n            <hr style="margin: 6px 0;">\n            <b>📧 Email:</b> {row.get('email') or 'N/A'}<br>\n            <b>📞 Tel:</b> {row.get('phone') or 'N/A'}<br>\n            <b>🌐 Web:</b> {row.get('website') or 'N/A'}\n        </div>\n        """
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=radius, color=color, weight=1.5, fill=True, fill_color=color, fill_opacity=fill_op, popup=folium.Popup(popup_html, max_width=360), tooltip=f"[{row['b2b_5d_score']:.2f} pts] {row['business_name']} ({row['workforce_scholarship_tier']})").add_to(group)
    fg_target.add_to(m)
    fg_rescued.add_to(m)
    fg_partial.add_to(m)
    fg_excluded.add_to(m)
    folium.Marker(location=[target_lat, target_lon], popup=f"<div style='font-family:Arial; min-width:205px;'><b style='font-size:14px; color:#d62728;'>☕ The Coffee — Bajío Hub</b><br><b>Hub de Intercepción Matutina (7:00–10:00 AM)</b><br>Coords: {target_lat:.6f}, {target_lon:.6f}</div>", tooltip='🎯 The Coffee — Bajío Hub (Hub Central)', icon=folium.Icon(color='red', icon='coffee', prefix='fa')).add_to(m)
    legend_html = '\n    <div style="position: fixed; bottom: 25px; left: 25px; width: 285px; background-color: rgba(255, 255, 255, 0.95);\n                border: 2px solid #333; z-index: 9999; font-size: 12px; padding: 12px; border-radius: 6px;\n                box-shadow: 0 4px 10px rgba(0,0,0,0.25); font-family: Arial, sans-serif;">\n        <b style="font-size: 13px;">🎯 OSINT 5D (INEGI/IMSS + SICT + AdaBoost)</b>\n        <hr style="margin: 6px 0;">\n        <span style="color:#1b7837; font-size:14px;">●</span> <b>Target Directo B2B</b> (Corp/Fin/Salud Priv.)<br>\n        <span style="color:#7b3294; font-size:14px;">●</span> <b>Rescatado</b> (Salario &ge;$24k &amp; &ge;35% Prof.)<br>\n        <span style="color:#e66101; font-size:14px;">●</span> <b>Target Parcial</b> (Manufactura/Hotel)<br>\n        <span style="color:#003399; font-size:14px;">●</span> <b>Estación Aforo SICT</b> (Flujo 7-10 AM)<br>\n        <span style="color:#d62728; font-weight:bold;">━━</span> <b>Corredor Commuter 7:00–10:00 AM</b><br>\n        <hr style="margin: 6px 0;">\n        <small style="color:#555;">Fuentes: INEGI DENUE &amp; Censos Económicos, IMSS SBC, SICT Datos Viales, OSRM, OpenStreetMap (ODbL).</small>\n    </div>\n    '
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f'✅ Interactive 5D OSINT Map saved -> {output_path}')
    return m
