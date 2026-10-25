"""
Unit tests for Stage 03 B2B Commercial Activation, Pitch Engine & Field Route Sequencing.
"""
import pytest
import polars as pl
from pathlib import Path


def generate_b2b_pitch(account: dict) -> dict:
    """
    Generates personalized multi-channel B2B pitches stratified by industry tier,
    academic tier, headcount, and contact risk guardrails.
    """
    name = account.get("canonical_parent_name") or account.get("primary_site_name") or "la empresa"
    status = account.get("b2b_status", "")
    academic_tier = account.get("workforce_academic_level", "corporate_professional")
    risk = account.get("contact_risk", "high")
    confidence = account.get("email_confidence", "none")
    channel = account.get("best_contact_channel", "")
    headcount = int(account.get("total_est_employees_midpoint") or 10)

    # 1. Determine Outreach Automation Guardrail
    if risk == "low" and confidence in ["high", "medium-high"]:
        guardrail = "automated_sequence_approved"
    elif risk == "medium":
        guardrail = "manual_approval_required"
    else:
        guardrail = "blocked_from_email_sequences"

    # 2. Industry / Academic Tailoring
    if academic_tier in ["high_automation_engineering", "specialized_technical"]:
        value_prop = "soluciones de abastecimiento de café de especialidad y coffee breaks corporativos para juntas operativas y cambios de turno"
        hook = f"Con más de {headcount:,} colaboradores en su complejo de Silao, sabemos que la productividad y la retención del talento técnico son críticas."
    elif academic_tier == "corporate_professional":
        value_prop = "convenios ejecutivos de barra de café y servicio para salas de consejo y atención a clientes corporativos"
        hook = f"Diseñado específicamente para optimizar la experiencia de los {headcount:,} profesionales en sus instalaciones."
    else:
        value_prop = "servicio ágil de café para oficinas administrativas y eventos corporativos"
        hook = f"Atendiendo el flujo operativo de {headcount:,} empleados con tiempos de entrega garantizados."

    subject = f"[Paseo MS] Convenio de Café & Coffee Breaks para {name}"
    
    body = (
        f"Estimado equipo de {name},\n\n"
        f"{hook}\n\n"
        f"Desde The Coffee — Paseo MS (Silao), ofrecemos {value_prop}, "
        f"con entrega programada a pie de planta o recepción ejecutiva.\n\n"
        f"¿Tendrían 5 minutos esta semana para presentarles el catálogo corporativo de degustación sin costo?\n\n"
        f"Saludos cordiales,\nEquipo Comercial — The Coffee Paseo MS"
    )

    linkedin = (
        f"Hola! Me pongo en contacto desde The Coffee Paseo MS. "
        f"Buscamos coordinar convenios de coffee break y abastecimiento para el equipo de {name}. "
        f"¿Podemos conectar?"
    )

    whatsapp = (
        f"Hola, buenas tardes. Me comunico de The Coffee en Paseo MS Silao. "
        f"Queremos enviar un kit de degustación de café de especialidad y catálogo de coffee breaks "
        f"para el área de compras/RH de {name}. ¿Con quién podríamos canalizarlo?"
    )

    return {
        "email_subject": subject,
        "email_body": body,
        "linkedin_pitch": linkedin,
        "whatsapp_script": whatsapp,
        "outreach_guardrail": guardrail,
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def solve_tsp_route_sequence(sites_df: pl.DataFrame, start_lat: float, start_lon: float) -> pl.DataFrame:
    """
    Solves optimal field-visit route sequence from start anchor using Nearest-Neighbor + 2-Opt.
    """
    records = sites_df.to_dicts()
    n = len(records)
    if n == 0:
        return sites_df

    # Nearest-Neighbor sequencing
    unvisited = list(range(n))
    curr_lat, curr_lon = start_lat, start_lon
    route = []

    while unvisited:
        best_idx = None
        best_dist = float("inf")
        for idx in unvisited:
            r = records[idx]
            d = haversine_km(curr_lat, curr_lon, float(r["lat"]), float(r["lon"]))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        route.append(best_idx)
        curr_lat = float(records[best_idx]["lat"])
        curr_lon = float(records[best_idx]["lon"])
        unvisited.remove(best_idx)

    # 2-Opt local search improvement
    def total_distance(order):
        d = haversine_km(start_lat, start_lon, float(records[order[0]]["lat"]), float(records[order[0]]["lon"]))
        for k in range(len(order) - 1):
            d += haversine_km(
                float(records[order[k]]["lat"]), float(records[order[k]]["lon"]),
                float(records[order[k+1]]["lat"]), float(records[order[k+1]]["lon"])
            )
        return d

    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if total_distance(new_route) < total_distance(route) - 1e-4:
                    route = new_route
                    improved = True

    # Build sequenced metrics
    ordered_records = []
    cum_km = 0.0
    c_lat, c_lon = start_lat, start_lon
    
    for seq_num, idx in enumerate(route, start=1):
        r = dict(records[idx])
        dest_lat, dest_lon = float(r["lat"]), float(r["lon"])
        leg_km = haversine_km(c_lat, c_lon, dest_lat, dest_lon)
        cum_km += leg_km
        travel_min = round(leg_km * 2.5) # Approx 25 km/h urban/industrial speed
        
        # Chronotype window
        chrono = r.get("chronotype", "Office_Standard_8AM")
        if chrono == "Early_Bird_6AM":
            window = "06:30 - 08:30 (Cambio Turno)"
        elif chrono == "Office_Standard_8AM":
            window = "09:00 - 11:30 (Horario Oficina)"
        else:
            window = "11:30 - 14:00 (Mediodía)"

        r["visit_order"] = seq_num
        r["leg_distance_km"] = round(leg_km, 2)
        r["cumulative_distance_km"] = round(cum_km, 2)
        r["estimated_travel_min"] = travel_min
        r["recommended_arrival_window"] = window
        ordered_records.append(r)
        c_lat, c_lon = dest_lat, dest_lon

    return pl.DataFrame(ordered_records)


# ==========================================
# TEST CASES
# ==========================================

def test_generate_b2b_pitch_engineering_low_risk():
    acc = {
        "canonical_parent_name": "AUMOVIO MEXICO",
        "workforce_academic_level": "high_automation_engineering",
        "contact_risk": "low",
        "email_confidence": "high",
        "total_est_employees_midpoint": 1225,
        "best_contact_channel": "Direct Email (Verified DENUE)",
    }
    pitch = generate_b2b_pitch(acc)
    assert "[Paseo MS]" in pitch["email_subject"]
    assert "AUMOVIO MEXICO" in pitch["email_subject"]
    assert "1,225" in pitch["email_body"]
    assert "productividad" in pitch["email_body"]
    assert pitch["outreach_guardrail"] == "automated_sequence_approved"
    assert len(pitch["linkedin_pitch"]) <= 300
    assert "Paseo MS" in pitch["whatsapp_script"]


def test_generate_b2b_pitch_synthetic_high_risk_blocked():
    acc = {
        "canonical_parent_name": "BANAMEX",
        "workforce_academic_level": "corporate_professional",
        "contact_risk": "high",
        "email_confidence": "low",
        "total_est_employees_midpoint": 40,
        "best_contact_channel": "Website / Synthesized Domain Email",
    }
    pitch = generate_b2b_pitch(acc)
    assert pitch["outreach_guardrail"] == "blocked_from_email_sequences"
    assert "salas de consejo" in pitch["email_body"]


def test_solve_tsp_route_sequence_optimization():
    # 4 locations along Highway 45 / FIPASI
    sites = pl.DataFrame({
        "denue_id": ["S1", "S2", "S3", "S4"],
        "business_name": ["PLANTA LEJOS", "PLANTA CERCA 1", "PLANTA CERCA 2", "PLANTA MEDIA"],
        "lat": [20.9900, 20.9550, 20.9580, 20.9700],
        "lon": [-101.4400, -101.4170, -101.4200, -101.4300],
        "chronotype": ["Office_Standard_8AM", "Early_Bird_6AM", "Office_Standard_8AM", "Early_Bird_6AM"]
    })
    start_lat, start_lon = 20.9538, -101.4161 # Paseo MS
    
    routed = solve_tsp_route_sequence(sites, start_lat, start_lon)
    
    assert len(routed) == 4
    assert routed["visit_order"].to_list() == [1, 2, 3, 4]
    
    # Nearest neighbor should visit close plants before far ones
    first_visited = routed["business_name"][0]
    last_visited = routed["business_name"][-1]
    assert first_visited in ["PLANTA CERCA 1", "PLANTA CERCA 2"]
    assert last_visited == "PLANTA LEJOS"
    
    # Cumulative distance must be strictly increasing
    cum_dists = routed["cumulative_distance_km"].to_list()
    assert cum_dists == sorted(cum_dists)
    assert cum_dists[-1] > cum_dists[0]
