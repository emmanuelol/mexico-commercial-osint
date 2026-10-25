import polars as pl

import math
ANCHOR_LAT, ANCHOR_LON = (21.1221, -101.6826)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = (math.radians(lat1), math.radians(lat2))
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def assign_offer_playbook(r: dict) -> dict:
    hc = int(r.get('total_est_employees_midpoint') or 10)
    scian = str(r.get('scian_code', ''))[:2]
    dist = float(r.get('distance_km') or 1.5)
    name = r.get('canonical_parent_name') or 'la empresa'
    if scian in ['31', '32', '33', '23'] or hc >= 100:
        code = 'Playbook A'
        name_offer = 'Convenio Corporativo & Coffee Box Juntas Matutinas (7:00–9:00 AM) vía Rappi con CFDI 4.0'
        pitch = f'Solución integral de coffee breaks y pedidos corporativos vía Rappi con facturación inmediata CFDI 4.0 para el equipo de {name}.'
    elif dist <= 2.0 and scian in ['52', '62', '54']:
        code = 'Playbook B'
        name_offer = 'Membresía Vecino Corporativo (15% Off 7–10 AM en App/Sucursal + Rappi B2B)'
        pitch = f'Convenio de descuento del 15% para el personal de {name} en compras presenciales/App (local_partners) y pedidos corporativos programados vía Rappi.'
    elif scian == '72':
        code = 'Playbook C'
        name_offer = 'Alianza Huésped Ejecutivo / Barra de Café Especialidad para Eventos'
        pitch = f'Alianza para abastecer coffee breaks en salones ejecutivos y paquetes de bienvenida para huéspedes VIP en {name}.'
    else:
        code = 'Playbook D'
        name_offer = 'Servicio Ágil de Café de Especialidad para Oficinas y Despachos'
        pitch = f'Suministro programado de café de grano y bebidas preparadas para la jornada matutina del equipo de {name}.'
    return {'playbook_code': code, 'playbook_name': name_offer, 'playbook_pitch': pitch}

def assign_route_corridor(lat: float, lon: float) -> str:
    d = haversine_km(ANCHOR_LAT, ANCHOR_LON, lat, lon)
    if d <= 1.0:
        return 'Ruta 1: Eje Central / Corredor Médico (<=1 km)'
    elif d <= 2.2:
        return 'Ruta 2: Corredor Financiero Centro (1-2 km)'
    else:
        return 'Ruta 3: Cluster Industrial Norte / Corredor Industrial (>2 km)'

def solve_route_tsp(subset_df: pl.DataFrame) -> pl.DataFrame:
    recs = subset_df.to_dicts()
    n = len(recs)
    if n == 0:
        return subset_df
    unvisited = list(range(n))
    curr_lat, curr_lon = (ANCHOR_LAT, ANCHOR_LON)
    order = []
    while unvisited:
        nxt = min(unvisited, key=lambda idx: haversine_km(curr_lat, curr_lon, float(recs[idx]['lat']), float(recs[idx]['lon'])))
        order.append(nxt)
        curr_lat, curr_lon = (float(recs[nxt]['lat']), float(recs[nxt]['lon']))
        unvisited.remove(nxt)

    def d_tot(od):
        td = haversine_km(ANCHOR_LAT, ANCHOR_LON, float(recs[od[0]]['lat']), float(recs[od[0]]['lon']))
        for i in range(len(od) - 1):
            td += haversine_km(float(recs[od[i]]['lat']), float(recs[od[i]]['lon']), float(recs[od[i + 1]]['lat']), float(recs[od[i + 1]]['lon']))
        return td
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if d_tot(candidate) < d_tot(order) - 0.0001:
                    order = candidate
                    improved = True
    ordered_out = []
    c_lat, c_lon = (ANCHOR_LAT, ANCHOR_LON)
    cum_d = 0.0
    for seq, idx in enumerate(order, start=1):
        item = dict(recs[idx])
        dest_lat, dest_lon = (float(item['lat']), float(item['lon']))
        leg = haversine_km(c_lat, c_lon, dest_lat, dest_lon)
        cum_d += leg
        item['route_stop_order'] = seq
        item['leg_km'] = round(leg, 2)
        item['cumulative_km'] = round(cum_d, 2)
        item['est_drive_min'] = round(leg * 2.5)
        ordered_out.append(item)
        c_lat, c_lon = (dest_lat, dest_lon)
    return pl.DataFrame(ordered_out)
