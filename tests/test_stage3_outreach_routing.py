"""
Unit tests for Stage 03 B2B Commercial Activation, Pitch Engine & Field Route Sequencing.
"""
import pytest
import polars as pl
from pathlib import Path


def generate_b2b_pitch(
    account: dict,
    anchor_name: str = "The Coffee — Bajío Hub",
    region_name: str = "el Bajío"
) -> dict:
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
        hook = f"Con más de {headcount:,} colaboradores en su complejo de {region_name}, sabemos que la productividad y la retención del talento técnico son críticas."
    elif academic_tier == "corporate_professional":
        value_prop = "convenios ejecutivos de barra de café y servicio para salas de consejo y atención a clientes corporativos"
        hook = f"Diseñado específicamente para optimizar la experiencia de los {headcount:,} profesionales en sus instalaciones."
    else:
        value_prop = "servicio ágil de café para oficinas administrativas y eventos corporativos"
        hook = f"Atendiendo el flujo operativo de {headcount:,} empleados con tiempos de entrega garantizados."

    subject = f"[{anchor_name}] Convenio de Café & Coffee Breaks para {name}"
    
    body = (
        f"Estimado equipo de {name},\n\n"
        f"{hook}\n\n"
        f"Desde {anchor_name}, ofrecemos {value_prop}, "
        f"con pedidos corporativos vía Rappi o entrega para juntas ejecutivas con facturación CFDI 4.0.\n\n"
        f"¿Tendrían 5 minutos esta semana para presentarles el catálogo corporativo de degustación sin costo?\n\n"
        f"Saludos cordiales,\nEquipo Comercial — {anchor_name}"
    )

    linkedin = (
        f"Hola! Me pongo en contacto desde {anchor_name}. "
        f"Buscamos coordinar convenios de coffee break y abastecimiento para el equipo de {name}. "
        f"¿Podemos conectar?"
    )

    whatsapp = (
        f"Hola, buenas tardes. Me comunico de {anchor_name}. "
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
    assert "[The Coffee — Bajío Hub]" in pitch["email_subject"]
    assert "AUMOVIO MEXICO" in pitch["email_subject"]
    assert "1,225" in pitch["email_body"]
    assert "productividad" in pitch["email_body"]
    assert pitch["outreach_guardrail"] == "automated_sequence_approved"
    assert len(pitch["linkedin_pitch"]) <= 300
    assert "Bajío Hub" in pitch["whatsapp_script"]


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
    start_lat, start_lon = 20.9538, -101.4161 # Target anchor hub
    
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


def normalize_stage3_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Standardizes English Stage-02 schema with bidirectional legacy Spanish fallback."""
    mapping = {
        "codigo_act": "scian_code",
        "per_ocu": "employee_strata",
        "correoelec": "email",
        "www": "website",
        "telefono": "phone",
        "nombre_estab": "business_name",
        "raz_social": "legal_name",
        "mun": "municipality",
    }
    rename_dict = {}
    for old_col, new_col in mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            rename_dict[old_col] = new_col
    res = df.rename(rename_dict) if rename_dict else df
    return res


def impute_morning_chronotype_4digit(scian_code: str | None, academic_level: str | None = None) -> str:
    """Imputes 7:00-10:00 AM coffee shift window using 4-digit SCIAN and academic tier."""
    code = str(scian_code or "").strip()
    prefix2 = code[:2]
    prefix3 = code[:3]
    prefix4 = code[:4]

    if prefix4 in ["3361", "3362", "3363", "5413"] or prefix2 in ["31", "32", "33", "23"] or prefix3 in ["484", "493"]:
        return "Turno_Ingenieria_630_730AM"
    if prefix3 in ["621", "622", "623"] or prefix2 == "62":
        return "Guardia_Medica_700_830AM"
    if prefix4 in ["5415", "5221", "7211"] or prefix2 in ["52", "55"] or academic_level == "corporate_professional":
        return "Ejecutivo_Oficina_800_930AM"
    return "Apertura_Comercial_900_1000AM"


def calculate_b2b_emv(
    headcount: int,
    b2b_5d_score: float,
    distance_km: float,
    chronotype: str,
    ticket_mxn: float = 65.0,
    group_order_ticket_mxn: float = 174.31,
    working_days: int = 20
) -> dict:
    """Calculates unified propensity (0-100) and Expected Monthly Coffee Revenue in MXN."""
    # Proximity decay factor
    decay = max(0.4, 1.0 - (distance_km / 8.0))
    # Chronotype alignment weight
    chrono_weights = {
        "Turno_Ingenieria_630_730AM": 0.25,
        "Ejecutivo_Oficina_800_930AM": 0.30,
        "Guardia_Medica_700_830AM": 0.28,
        "Apertura_Comercial_900_1000AM": 0.15,
    }
    capture_rate = min(0.35, max(0.05, chrono_weights.get(chronotype, 0.15) * decay))
    
    # White collar headcount estimate
    white_collar_est = max(1, round(headcount * 0.40)) if headcount > 50 else max(1, headcount)
    monthly_rev_mxn = round(white_collar_est * capture_rate * ticket_mxn * working_days, 2)
    
    # Normalized Propensity Score 0 to 100
    propensity_score = round(min(100.0, max(10.0, (b2b_5d_score * 8.5) + (capture_rate * 80.0))), 1)
    
    return {
        "b2b_propensity_score": propensity_score,
        "capture_rate": round(capture_rate, 3),
        "est_monthly_coffee_rev_mxn": monthly_rev_mxn,
    }


def assign_offer_playbook(account: dict) -> str:
    """Assigns the 4-tier B2B Coffee Offer Playbook."""
    hc = int(account.get("total_est_employees_midpoint") or 10)
    scian = str(account.get("scian_code", ""))[:2]
    dist = float(account.get("distance_km") or 1.5)
    
    if (scian in ["31", "32", "33", "23"] or hc >= 100):
        return "Playbook A: Convenio Corporativo & Coffee Box Juntas Matutinas (7:00-9:00 AM) vía Rappi con CFDI 4.0"
    elif dist <= 2.0 and scian in ["52", "62", "54"]:
        return "Playbook B: Membresía Vecino Corporativo (15% Off 7-10 AM en App/Sucursal + Rappi B2B)"
    elif scian == "72":
        return "Playbook C: Alianza Huésped Ejecutivo / Barra de Café Especialidad para Eventos"
    else:
        return "Playbook D: Servicio Ágil de Café de Especialidad para Oficinas y Despachos"


def test_bidirectional_schema_normalizer():
    # Spanish INEGI legacy schema
    df_spanish = pl.DataFrame({
        "codigo_act": ["336110"],
        "per_ocu": ["251 y más personas"],
        "correoelec": ["planta@auto.com"],
        "www": ["www.auto.com"],
        "telefono": ["4721234567"],
        "nombre_estab": ["PLANTA AUTOMOTRIZ"],
        "raz_social": ["AUTOMOTRIZ DEL BAJIO SA DE CV"],
        "mun": ["Leon de los Aldama"]
    })
    normalized = normalize_stage3_schema(df_spanish)
    assert "scian_code" in normalized.columns
    assert "employee_strata" in normalized.columns
    assert "email" in normalized.columns
    assert "website" in normalized.columns
    assert "phone" in normalized.columns
    assert "business_name" in normalized.columns
    assert "legal_name" in normalized.columns
    assert "municipality" in normalized.columns
    assert normalized["scian_code"][0] == "336110"


def test_4digit_scian_morning_chronotype_imputer():
    assert impute_morning_chronotype_4digit("336110") == "Turno_Ingenieria_630_730AM"
    assert impute_morning_chronotype_4digit("541330") == "Turno_Ingenieria_630_730AM"
    assert impute_morning_chronotype_4digit("622110") == "Guardia_Medica_700_830AM"
    assert impute_morning_chronotype_4digit("541510") == "Ejecutivo_Oficina_800_930AM"
    assert impute_morning_chronotype_4digit("522110") == "Ejecutivo_Oficina_800_930AM"
    assert impute_morning_chronotype_4digit("721111") == "Ejecutivo_Oficina_800_930AM"
    assert impute_morning_chronotype_4digit("461110") == "Apertura_Comercial_900_1000AM"


def test_b2b_propensity_and_monthly_revenue_emv():
    emv_res = calculate_b2b_emv(
        headcount=350,
        b2b_5d_score=8.5,
        distance_km=1.2,
        chronotype="Turno_Ingenieria_630_730AM",
        ticket_mxn=65.0,
        working_days=20
    )
    assert emv_res["b2b_propensity_score"] >= 80.0
    assert emv_res["est_monthly_coffee_rev_mxn"] > 5000.0


def test_b2b_offer_playbook_assignment():
    acc_plant = {"total_est_employees_midpoint": 350, "scian_code": "3361", "distance_km": 3.5}
    acc_bank = {"total_est_employees_midpoint": 25, "scian_code": "5221", "distance_km": 0.8}
    acc_hotel = {"total_est_employees_midpoint": 40, "scian_code": "7211", "distance_km": 1.1}
    acc_office = {"total_est_employees_midpoint": 10, "scian_code": "5611", "distance_km": 3.0}

    assert "Playbook A" in assign_offer_playbook(acc_plant)
    assert "Playbook B" in assign_offer_playbook(acc_bank)
    assert "Playbook C" in assign_offer_playbook(acc_hotel)
    assert "Playbook D" in assign_offer_playbook(acc_office)

