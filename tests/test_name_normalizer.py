import math
import re
import unicodedata
import hashlib
from pathlib import Path
import polars as pl
import pytest
from rapidfuzz import fuzz, process

ROOT_DIR = Path(__file__).resolve().parent.parent

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b("
    r"S\.?\s*A\.?\s*P\.?\s*I\.?\s*(?:DE\s*C\.?\s*V\.?)?|"
    r"S\.?\s*A\.?\s*B\.?\s*(?:DE\s*C\.?\s*V\.?)?|"
    r"S\.?\s*A\.?\s*DE\s*C\.?\s*V\.?|"
    r"S\.?\s*DE\s*R\.?\s*L\.?\s*(?:DE\s*C\.?\s*V\.?)?|"
    r"S\.?\s*C\.?\s*DE\s*R\.?\s*L\.?|"
    r"S\.?\s*C\.?|"
    r"A\.?\s*C\.?|"
    r"S\.?\s*A\.?|"
    r"SAB\s*DE\s*CV"
    r")\b",
    re.IGNORECASE
)

BRANCH_QUALIFIER_REGEX = re.compile(
    r"\b(PLANTA|MATRIZ|SUCURSAL|CORPORATIVO|CAMPUS|NORTE|LEON|GUANAJUATO|BAJIO|MEXICO)\b",
    re.IGNORECASE
)

PUNCTUATION_REGEX = re.compile(r"[,.\-'\"]")


def normalize_mexican_legal_name(name: str) -> str:
    """Normalizes a Mexican company or trade name: uppercase, accent removal, legal suffix stripping."""
    if not name or not isinstance(name, str):
        return ""
    # 1. Strip accents via NFKD
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).upper()
    
    # 2. Strip legal suffixes
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)
    
    # 3. Strip punctuation
    text = PUNCTUATION_REGEX.sub(" ", text)
    
    # 4. Collapse whitespace
    text = " ".join(text.split()).strip()
    return text


def strip_corporate_branch_qualifiers(name: str) -> str:
    """Strips secondary branch/plant descriptors (e.g., PLANTA, MATRIZ, NORTE) for campus parent resolution."""
    cleaned = normalize_mexican_legal_name(name)
    cleaned = BRANCH_QUALIFIER_REGEX.sub(" ", cleaned)
    cleaned = PUNCTUATION_REGEX.sub(" ", cleaned)
    return " ".join(cleaned.split()).strip()


def encode_geohash_manual(latitude: float, longitude: float, precision: int = 6) -> str:
    """Vector-friendly standard geohash encoder in pure Python without external C libraries."""
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2.0
            if longitude > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2.0
            if latitude > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(base32[ch])
            bit = 0
            ch = 0
    return "".join(geohash)


def assign_geohash_block(df: pl.DataFrame, precision: int = 6) -> pl.DataFrame:
    geohashes = [
        encode_geohash_manual(lat, lon, precision=precision)
        for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list())
    ]
    return df.with_columns(pl.Series("geohash", geohashes))


def assign_entity_ids(df: pl.DataFrame) -> pl.DataFrame:
    entity_ids = []
    for norm_name, gh in zip(df["norm_name"].to_list(), df["geohash"].to_list()):
        raw_key = f"{norm_name}_{gh[:4]}".encode("utf-8")
        entity_ids.append(f"ent_{hashlib.sha1(raw_key).hexdigest()[:12]}")
    return df.with_columns(pl.Series("entity_id", entity_ids))


def spatial_join_osm_context(
    denue_df: pl.DataFrame,
    osm_df: pl.DataFrame,
    max_dist_m: float = 50.0,
) -> pl.DataFrame:
    """Cross-source spatial resolution between DENUE and OSM amenities respecting ODbL boundaries."""
    if len(osm_df) == 0:
        return denue_df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("nearest_osm_id"),
            pl.lit(None).cast(pl.Float64).alias("osm_match_dist_m"),
            pl.lit(False).alias("osm_verified_storefront"),
            pl.lit("").alias("osm_poi_context"),
        ])

    osm_records = osm_df.select(["osm_id", "lat", "lon", "name", "amenity", "office", "shop"]).to_dicts()
    r_earth_m = 6371000.0
    rad = math.pi / 180.0

    nearest_ids = []
    nearest_dists = []
    verified_flags = []
    contexts = []

    for row in denue_df.iter_rows(named=True):
        d_lat, d_lon = row["lat"], row["lon"]
        best_dist = float("inf")
        best_osm = None

        for osm in osm_records:
            o_lat, o_lon = osm["lat"], osm["lon"]
            dlat = (o_lat - d_lat) * rad
            dlon = (o_lon - d_lon) * rad
            a = math.sin(dlat / 2.0) ** 2 + math.cos(d_lat * rad) * math.cos(o_lat * rad) * math.sin(dlon / 2.0) ** 2
            dist_m = 2.0 * math.asin(math.sqrt(a)) * r_earth_m

            if dist_m < best_dist:
                best_dist = dist_m
                best_osm = osm

        if best_osm and best_dist <= 250.0:
            nearest_ids.append(str(best_osm["osm_id"]))
            nearest_dists.append(round(best_dist, 1))
            verified_flags.append(best_dist <= max_dist_m)
            ctx = best_osm.get("name") or best_osm.get("amenity") or best_osm.get("office") or best_osm.get("shop") or ""
            contexts.append(str(ctx))
        else:
            nearest_ids.append(None)
            nearest_dists.append(None)
            verified_flags.append(False)
            contexts.append("")

    return denue_df.with_columns([
        pl.Series("nearest_osm_id", nearest_ids, dtype=pl.Utf8),
        pl.Series("osm_match_dist_m", nearest_dists, dtype=pl.Float64),
        pl.Series("osm_verified_storefront", verified_flags, dtype=pl.Boolean),
        pl.Series("osm_poi_context", contexts, dtype=pl.Utf8),
    ])


def classify_b2b_email_role(email: str | None) -> str:
    """Classifies B2B email into decision maker, functional generic inbox, or personal/free email."""
    if not email or not isinstance(email, str) or "@" not in email:
        return "none"
    clean_em = email.strip().lower()
    free_domains = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "live.com", "prodigy.net.mx"]
    if any(clean_em.endswith(f"@{d}") for d in free_domains):
        return "free_personal"
    
    user_part = clean_em.split("@")[0]
    decision_makers = ["director", "gerente", "rh", "recursoshumanos", "purchasing", "compras", "talento", "ceo", "cfo"]
    generic_roles = ["contacto", "info", "recepcion", "atencion", "ventas", "servicio", "facturacion"]
    
    if any(dm in user_part for dm in decision_makers):
        return "decision_maker_role"
    if any(gen in user_part for gen in generic_roles):
        return "generic_role_inbox"
    return "corporate_direct"


def cluster_corporate_entities(df: pl.DataFrame, fuzzy_threshold: float = 0.85) -> pl.DataFrame:
    """
    Clusters multi-plant corporate campuses and duplicate branch entries using
    RapidFuzz WRatio and branch-qualifier stripping across legal and commercial names.
    Prevents false merges on empty or generic legal names and propagates contacts.
    """
    name_col = "nombre_estab" if "nombre_estab" in df.columns else "business_name"
    legal_col = "raz_social" if "raz_social" in df.columns else "legal_name"
    id_col = "clee" if "clee" in df.columns else ("id" if "id" in df.columns else "row_id")

    if "geohash" not in df.columns:
        df = assign_geohash_block(df, precision=6)

    norm_commercial = [normalize_mexican_legal_name(str(n or "")) for n in df[name_col].to_list()]
    parent_commercial = [strip_corporate_branch_qualifiers(str(n or "")) for n in df[name_col].to_list()]
    norm_legal = [normalize_mexican_legal_name(str(l or "")) if l else "" for l in df[legal_col].to_list()]
    parent_legal = [strip_corporate_branch_qualifiers(str(l or "")) if l else "" for l in df[legal_col].to_list()]

    df_prep = df.with_columns([
        pl.Series("norm_name", norm_commercial),
        pl.Series("parent_name", parent_commercial),
        pl.Series("norm_legal", norm_legal),
        pl.Series("parent_legal", parent_legal),
    ])

    generic_stop = {"SERVICIOS", "COMERCIAL", "GRUPO", "DISTRIBUIDORA", "LOGISTICA", "CENTRO", "DESPACHO", "TIENDA", "ABARROTES"}
    records = df_prep.to_dicts()
    n = len(records)

    parent_map = list(range(n))

    def find(i):
        if parent_map[i] == i:
            return i
        parent_map[i] = find(parent_map[i])
        return parent_map[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent_map[root_j] = root_i

    FREE_DOMAINS = {"gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "live.com", "live.com.mx", "prodigy.net.mx", "gob.mx"}

    for i in range(n):
        ri = records[i]
        legi = (ri.get("parent_legal") or ri.get("norm_legal") or "").strip().upper()
        pcommi = (ri.get("parent_name") or ri.get("norm_name") or "").strip().upper()
        emi = (ri.get("email") or ri.get("correoelec") or "").strip().lower()
        domi = emi.split("@")[1] if ("@" in emi and emi.split("@")[1] not in FREE_DOMAINS) else ""

        for j in range(i + 1, n):
            rj = records[j]
            legj = (rj.get("parent_legal") or rj.get("norm_legal") or "").strip().upper()
            pcommj = (rj.get("parent_name") or rj.get("norm_name") or "").strip().upper()
            emj = (rj.get("email") or rj.get("correoelec") or "").strip().lower()
            domj = emj.split("@")[1] if ("@" in emj and emj.split("@")[1] not in FREE_DOMAINS) else ""

            match = False
            # 1. Exact parent legal or parent commercial name match
            if legi and legj and legi == legj and len(legi) >= 5 and legi not in generic_stop:
                match = True
            elif pcommi and pcommj and pcommi == pcommj and len(pcommi) >= 5 and pcommi not in generic_stop:
                match = True
            # 2. Shared corporate email or corporate domain (merges Lear, Bodycote, Hotsson/Milenium)
            elif emi and emj and emi == emj:
                match = True
            elif domi and domj and domi == domj and not any(k in domi for k in ["edu.mx", "org.mx"]):
                match = True
            # 3. High-confidence RapidFuzz match on stripped parent legal names
            elif legi and legj and len(legi) >= 6 and len(legj) >= 6 and (fuzz.WRatio(legi, legj) / 100.0) >= fuzzy_threshold:
                match = True

            if match:
                union(i, j)

    from collections import defaultdict
    clusters = defaultdict(list)
    for i in range(n):
        root = find(i)
        clusters[root].append(i)

    canonical_entity_ids = [None] * n
    canonical_parent_names = [None] * n
    is_duplicate = [False] * n
    inherited_emails = [None] * n
    inherited_websites = [None] * n
    inherited_phones = [None] * n

    for root, indices in clusters.items():
        def site_priority(idx):
            r = records[idx]
            name = str(r.get(name_col, "")).upper()
            is_mat = 1 if "MATRIZ" in name or "CORPORATIVO" in name else 0
            has_em = 1 if r.get("email") or r.get("correoelec") else 0
            score = float(r.get("b2b_5d_score") or 0.0)
            return (is_mat, has_em, score)

        sorted_indices = sorted(indices, key=site_priority, reverse=True)
        primary_idx = sorted_indices[0]
        primary_rec = records[primary_idx]

        best_legal = primary_rec.get("raz_social") or primary_rec.get("legal_name")
        cleaned_leg = normalize_mexican_legal_name(best_legal) if best_legal else ""
        if cleaned_leg and len(cleaned_leg) >= 5 and cleaned_leg not in generic_stop:
            canonical_name = cleaned_leg
        else:
            canonical_name = primary_rec.get("parent_name") or primary_rec.get(name_col)

        canonical_id = f"ent_{hashlib.sha1(canonical_name.encode('utf-8')).hexdigest()[:12]}"

        best_email = ""
        best_website = ""
        best_phone = ""
        for idx in sorted_indices:
            r = records[idx]
            em = r.get("email") or r.get("correoelec") or ""
            web = r.get("website") or r.get("www") or ""
            ph = r.get("phone") or r.get("telefono") or ""
            if em and not best_email:
                best_email = str(em).strip()
            if web and not best_website:
                best_website = str(web).strip()
            if ph and not best_phone:
                best_phone = str(ph).strip()

        for idx in sorted_indices:
            canonical_entity_ids[idx] = canonical_id
            canonical_parent_names[idx] = canonical_name
            if idx != primary_idx:
                is_duplicate[idx] = True

            curr_em = records[idx].get("email") or records[idx].get("correoelec") or ""
            curr_web = records[idx].get("website") or records[idx].get("www") or ""
            curr_ph = records[idx].get("phone") or records[idx].get("telefono") or ""

            inherited_emails[idx] = best_email if (not curr_em and best_email) else curr_em
            inherited_websites[idx] = best_website if (not curr_web and best_website) else curr_web
            inherited_phones[idx] = best_phone if (not curr_ph and best_phone) else curr_ph

    df_out = df_prep.with_columns([
        pl.Series("canonical_entity_id", canonical_entity_ids),
        pl.Series("canonical_parent_name", canonical_parent_names),
        pl.Series("is_duplicate", is_duplicate),
        pl.Series("inherited_email", inherited_emails),
        pl.Series("inherited_website", inherited_websites),
        pl.Series("inherited_phone", inherited_phones),
    ])

    site_counts = df_out.group_by("canonical_entity_id").agg(pl.len().alias("campus_site_count"))
    df_out = df_out.join(site_counts, on="canonical_entity_id", how="left")

    payroll_col = "estimated_monthly_payroll_mxn" if "estimated_monthly_payroll_mxn" in df_out.columns else (
        "est_monthly_payroll_mxn" if "est_monthly_payroll_mxn" in df_out.columns else None
    )
    if payroll_col:
        payroll_agg = df_out.group_by("canonical_entity_id").agg(pl.col(payroll_col).sum().alias("campus_consolidated_payroll_mxn"))
        df_out = df_out.join(payroll_agg, on="canonical_entity_id", how="left")
    else:
        df_out = df_out.with_columns(pl.lit(0.0).alias("campus_consolidated_payroll_mxn"))

    return df_out


def synthesize_corporate_domain_contact(website: str | None) -> dict:
    """Derives sanitized corporate domain and standard B2B functional role inboxes."""
    if not website or not isinstance(website, str):
        return {"domain": "", "synthesized_email": "", "alternate_inboxes": []}

    clean_w = website.strip().lower()
    for prefix in ["https://", "http://", "www."]:
        if clean_w.startswith(prefix):
            clean_w = clean_w[len(prefix):]
    clean_w = clean_w.split("/")[0].split("?")[0].strip()

    free_domains = ["facebook.com", "instagram.com", "twitter.com", "google.com", "maps.google.com"]
    if not clean_w or "." not in clean_w or any(clean_w.endswith(fd) for fd in free_domains):
        return {"domain": "", "synthesized_email": "", "alternate_inboxes": []}

    domain = clean_w
    synth_email = f"contacto@{domain}"
    alternates = [f"rh@{domain}", f"compras@{domain}", f"ventas@{domain}"]
    return {
        "domain": domain,
        "synthesized_email": synth_email,
        "alternate_inboxes": alternates,
    }


def evaluate_stage2_actionability(rec: dict) -> dict:
    """Evaluates multi-channel B2B actionability across direct email, campus inheritance, domain synthesis, and phone/ABM."""
    b2b_status = rec.get("b2b_status", "")
    is_rescued = rec.get("is_rescued_partial", False)
    email = (rec.get("email") or "").strip()
    inherited_em = (rec.get("inherited_email") or "").strip()
    website = (rec.get("website") or "").strip()
    phone = (rec.get("phone") or "").strip()
    score = float(rec.get("b2b_5d_score") or 0.0)
    headcount = int(rec.get("estimated_headcount") or 0)

    # 1. Direct Email (Target or Rescued Partial with verified direct email)
    if (b2b_status == "target" or is_rescued) and email:
        return {
            "is_actionable": True,
            "channel": "Direct Email (Verified DENUE)",
            "rescue_reason": "Stage-01 Verified Direct Email",
            "effective_email": email,
            "actionable_tier": "Tier-1 Direct Email",
        }

    # 2. Campus Inherited Email
    if (b2b_status == "target" or is_rescued) and inherited_em:
        return {
            "is_actionable": True,
            "channel": "Campus Inherited Email (Parent Entity)",
            "rescue_reason": "Campus Contact Inheritance",
            "effective_email": inherited_em,
            "actionable_tier": "Tier-1 Campus Inherited",
        }

    # 3. Target without email, but has corporate website -> Domain Synthesized
    synth = synthesize_corporate_domain_contact(website)
    if (b2b_status == "target" or is_rescued) and synth["synthesized_email"]:
        return {
            "is_actionable": True,
            "channel": "Website / Synthesized Domain Email",
            "rescue_reason": "Corporate Web Domain Contact Synthesis",
            "effective_email": synth["synthesized_email"],
            "actionable_tier": "Tier-2 Domain Synthesized",
        }

    # 4. Tier-1 Corporate Domain Partial Upgrade
    if b2b_status == "partial" and score >= 4.8:
        role = classify_b2b_email_role(email)
        if email and role in ["corporate_direct", "decision_maker_role", "generic_role_inbox"]:
            return {
                "is_actionable": True,
                "channel": "Direct Email (Rescued Tier-1 Corporate)",
                "rescue_reason": "Tier-1 Corporate Domain Partial Upgrade",
                "effective_email": email,
                "actionable_tier": "Tier-1 Rescued Partial",
            }
        elif website and synth["synthesized_email"] and headcount >= 31:
            return {
                "is_actionable": True,
                "channel": "Website / Synthesized Domain Email",
                "rescue_reason": "Tier-1 Corporate Headcount & Domain Partial Upgrade",
                "effective_email": synth["synthesized_email"],
                "actionable_tier": "Tier-2 Rescued Partial",
            }

    # 5. Direct Phone + LinkedIn ABM
    if (b2b_status == "target" or is_rescued) and phone:
        return {
            "is_actionable": True,
            "channel": "Direct Phone + LinkedIn ABM",
            "rescue_reason": "Multi-Channel Phone & Executive Search Rescue",
            "effective_email": "",
            "actionable_tier": "Tier-3 Phone / ABM",
        }

    # 6. Field Sales + LinkedIn ABM
    if b2b_status == "target" or is_rescued:
        return {
            "is_actionable": True,
            "channel": "Field Sales + LinkedIn ABM",
            "rescue_reason": "Storefront Physical Visit & Executive Search Rescue",
            "effective_email": "",
            "actionable_tier": "Tier-4 Field Sales / ABM",
        }

    return {
        "is_actionable": False,
        "channel": "Non-Actionable (Low Priority)",
        "rescue_reason": "Does not meet commercial qualification criteria",
        "effective_email": "",
        "actionable_tier": "Inactive",
    }


STRATA_MIDPOINT = {
    "0 a 5 personas": 3,
    "6 a 10 personas": 8,
    "11 a 30 personas": 20,
    "31 a 50 personas": 40,
    "51 a 100 personas": 75,
    "101 a 250 personas": 175,
    "251 y más personas": 350,
}


def map_employee_strata_midpoint(strata: str | None) -> int:
    """Exact mapping of DENUE employee strata to realistic headcount midpoint."""
    if not strata or not isinstance(strata, str):
        return 0
    clean = strata.strip().lower()
    for key, val in STRATA_MIDPOINT.items():
        if key in clean or clean in key:
            return val
    if "251" in clean:
        return 350
    if "101" in clean:
        return 175
    if "51" in clean:
        return 75
    if "31" in clean:
        return 40
    if "11" in clean:
        return 20
    if "6" in clean:
        return 8
    if "0" in clean or "1" in clean:
        return 3
    return 0


def evaluate_contact_confidence(contact: str | None, source: str = "direct_denue") -> dict:
    """Assigns rigorous provenance, confidence score, and risk flags to B2B contacts."""
    if not contact or not isinstance(contact, str) or contact.strip() in ["", "-", "none"]:
        return {
            "email_source": "none",
            "email_confidence": "none",
            "contact_verified": False,
            "contact_risk": "low",
        }
    if source == "direct_denue":
        return {
            "email_source": "direct_denue",
            "email_confidence": "high",
            "contact_verified": True,
            "contact_risk": "low",
        }
    if source == "scraped_website":
        return {
            "email_source": "scraped_website",
            "email_confidence": "medium-high",
            "contact_verified": True,
            "contact_risk": "low",
        }
    if source == "campus_inherited":
        return {
            "email_source": "campus_inherited",
            "email_confidence": "medium",
            "contact_verified": False,
            "contact_risk": "medium",
        }
    if source == "synthesized_domain":
        return {
            "email_source": "synthesized_domain",
            "email_confidence": "low",
            "contact_verified": False,
            "contact_risk": "high",
        }
    return {
        "email_source": source,
        "email_confidence": "low",
        "contact_verified": False,
        "contact_risk": "medium",
    }


def classify_workforce_academic_level(wage_mxn: float | None, degree_share: float | None) -> str:
    """Classifies establishment workforce academic / technical buyer tier."""
    wage = float(wage_mxn or 0.0)
    share = float(degree_share or 0.0)
    if share >= 0.60 and wage >= 25000:
        return "high_automation_engineering"
    elif share >= 0.35 and wage >= 18000:
        return "corporate_professional"
    elif share >= 0.20 or wage >= 15000:
        return "specialized_technical"
    else:
        return "traditional_labor"


def build_canonical_action_master(df_sites: pl.DataFrame) -> pl.DataFrame:
    """Builds strictly deduplicated 1-row-per-corporate-entity master outreach table."""
    act = df_sites.filter(pl.col("is_stage2_actionable") == True) if "is_stage2_actionable" in df_sites.columns else df_sites
    
    payroll_col = "est_monthly_payroll_mxn" if "est_monthly_payroll_mxn" in act.columns else (
        "estimated_monthly_payroll_mxn" if "estimated_monthly_payroll_mxn" in act.columns else None
    )

    agg_exprs = [
        pl.len().alias("site_count"),
        pl.col("canonical_parent_name").first().alias("canonical_parent_name"),
        pl.col("business_name").first().alias("primary_site_name"),
        pl.col("b2b_status").first().alias("b2b_status"),
        pl.col("b2b_5d_score").max().alias("max_b2b_5d_score"),
        pl.col("est_employees_midpoint").sum().alias("total_est_employees_midpoint") if "est_employees_midpoint" in act.columns else pl.lit(0).alias("total_est_employees_midpoint"),
        (pl.col(payroll_col).sum().alias("consolidated_payroll_mxn") if payroll_col else pl.lit(0.0).alias("consolidated_payroll_mxn")),
        pl.col("actionable_channel").first().alias("best_contact_channel"),
        pl.col("effective_contact").first().alias("effective_contact"),
        pl.col("email_source").first().alias("email_source") if "email_source" in act.columns else pl.lit("unknown").alias("email_source"),
        pl.col("email_confidence").first().alias("email_confidence") if "email_confidence" in act.columns else pl.lit("low").alias("email_confidence"),
    ]

    return act.group_by("canonical_entity_id").agg(agg_exprs).sort("max_b2b_5d_score", descending=True)


def build_site_action_list(df_sites: pl.DataFrame) -> pl.DataFrame:
    """Builds physical site-level routing table for field visits and maps."""
    act = df_sites.filter(pl.col("is_stage2_actionable") == True) if "is_stage2_actionable" in df_sites.columns else df_sites
    return act.with_columns([
        pl.when(pl.col("is_duplicate") == False)
        .then(pl.lit("primary_site"))
        .otherwise(pl.lit("branch_secondary_site"))
        .alias("role_in_campus")
    ])


def test_normalize_mexican_legal_suffixes():
    test_cases = [
        ("GRUPO INDUSTRIAL DEL BAJÍO S.A. DE C.V.", "GRUPO INDUSTRIAL DEL BAJIO"),
        ("TECNOLOGIA Y SOFTWARE S.A.P.I. DE C.V.", "TECNOLOGIA Y SOFTWARE"),
        ("LOGISTICA NACIONAL S. DE R.L. DE C.V.", "LOGISTICA NACIONAL"),
        ("SERVICIOS CONTABLES Y FISCALES S.C.", "SERVICIOS CONTABLES Y FISCALES"),
        ("FUNDACION COMUNITARIA A.C.", "FUNDACION COMUNITARIA"),
        ("COMERCIALIZADORA AGRICOLA S.A.", "COMERCIALIZADORA AGRICOLA"),
        ("COOPERATIVA DE TRANSPORTE S.C. DE R.L.", "COOPERATIVA DE TRANSPORTE"),
        ("BANCO MERCANTIL SAB DE CV", "BANCO MERCANTIL"),
        ("CORPORATIVO NACIONAL, S.A. B. DE C.V.", "CORPORATIVO NACIONAL"),
        ("FARMACIAS DE MÉXICO, S.A. DE C.V.", "FARMACIAS DE MEXICO"),
    ]
    for raw, expected in test_cases:
        result = normalize_mexican_legal_name(raw)
        assert result == expected, f"Failed on '{raw}': got '{result}', expected '{expected}'"


def test_strip_corporate_branch_qualifiers():
    assert strip_corporate_branch_qualifiers("AUMOVIO MEXICO PLANTA NORTE S.A. DE C.V.") == "AUMOVIO"
    assert strip_corporate_branch_qualifiers("AUMOVIO MEXICO MATRIZ") == "AUMOVIO"
    assert strip_corporate_branch_qualifiers("SEGLO OPERACIONES LOGISTICAS NORTE") == "SEGLO OPERACIONES LOGISTICAS"


def test_geohash_assignment():
    df = pl.DataFrame({
        "lat": [21.1221, 21.1230],
        "lon": [-101.6826, -101.6830],
        "norm_name": ["EMPRESA UNO", "EMPRESA DOS"]
    })
    df_gh = assign_geohash_block(df, precision=6)
    assert "geohash" in df_gh.columns
    assert len(df_gh["geohash"][0]) == 6
    # Coordinates ~100m apart should share geohash prefix 5
    assert df_gh["geohash"][0][:5] == df_gh["geohash"][1][:5]


def test_entity_id_generation():
    df = pl.DataFrame({
        "lat": [21.1221],
        "lon": [-101.6826],
        "norm_name": ["TECNOLOGIA DEL CENTRO"],
        "geohash": ["9exyzz"]
    })
    df_ent = assign_entity_ids(df)
    assert "entity_id" in df_ent.columns
    assert df_ent["entity_id"][0].startswith("ent_")


def test_fuzzy_matching_rapidfuzz():
    name_a = "DISTRIBUIDORA DE ACEROS MEXICANA"
    name_b = "DISTRIBUIDORA DE ACEROS DE MEXICO"
    score = fuzz.WRatio(name_a, name_b) / 100.0
    assert score > 0.85


def test_spatial_join_osm_context():
    denue = pl.DataFrame({
        "id": ["1", "2"],
        "lat": [20.9538, 20.9990],
        "lon": [-101.4161, -101.5000],
    })
    osm = pl.DataFrame({
        "osm_id": ["osm_001"],
        "lat": [20.95382],
        "lon": [-101.41608],
        "name": ["Plaza Paseo"],
        "amenity": ["cafe"],
        "office": [""],
        "shop": [""],
    })
    joined = spatial_join_osm_context(denue, osm, max_dist_m=50.0)
    assert "nearest_osm_id" in joined.columns
    assert "osm_verified_storefront" in joined.columns
    assert joined["osm_verified_storefront"][0] is True
    assert joined["osm_verified_storefront"][1] is False
    assert joined["nearest_osm_id"][0] == "osm_001"
    assert joined["nearest_osm_id"][1] is None


def test_classify_b2b_email_role():
    assert classify_b2b_email_role("rh@continental.com") == "decision_maker_role"
    assert classify_b2b_email_role("compras.planta@aumovio.com") == "decision_maker_role"
    assert classify_b2b_email_role("contacto@empresa.com.mx") == "generic_role_inbox"
    assert classify_b2b_email_role("juan.perez@continental.com") == "corporate_direct"
    assert classify_b2b_email_role("negocio@gmail.com") == "free_personal"
    assert classify_b2b_email_role(None) == "none"


def test_corporate_clustering_and_inheritance():
    df = pl.DataFrame({
        "clee": ["110370001", "110370002", "110370003"],
        "nombre_estab": ["AUMOVIO MATRIZ", "AUMOVIO BODEGA", "PAPELERIA LOPEZ"],
        "raz_social": ["AUMOVIO MEXICO S.A. DE C.V.", "AUMOVIO MEXICO S.A. DE C.V.", ""],
        "lat": [20.95, 20.952, 20.951],
        "lon": [-101.41, -101.412, -101.411],
        "email": ["contacto@aumovio.com", "", ""],
        "phone": ["4721234567", "", "4729876543"],
        "website": ["www.aumovio.com", "", ""],
        "b2b_status": ["target", "target", "low_potential"],
        "b2b_5d_score": [8.5, 8.2, 3.1],
        "estimated_headcount": [250, 40, 3],
        "estimated_monthly_payroll_mxn": [2500000.0, 400000.0, 30000.0],
    })
    clustered = cluster_corporate_entities(df)
    # Check that AUMOVIO records are clustered together
    aumovio_1 = clustered.filter(pl.col("clee") == "110370001").to_dicts()[0]
    aumovio_2 = clustered.filter(pl.col("clee") == "110370002").to_dicts()[0]
    pape = clustered.filter(pl.col("clee") == "110370003").to_dicts()[0]

    assert aumovio_1["canonical_entity_id"] == aumovio_2["canonical_entity_id"]
    assert aumovio_2["canonical_parent_name"] == "AUMOVIO MEXICO"
    assert aumovio_2["campus_site_count"] == 2
    # Campus contact inheritance: AUMOVIO BODEGA inherits email and website from MATRIZ
    assert aumovio_2["inherited_email"] == "contacto@aumovio.com"
    # Papeleria Lopez with empty raz_social must NOT be falsely merged into AUMOVIO!
    assert pape["canonical_entity_id"] != aumovio_1["canonical_entity_id"]
    assert pape["campus_site_count"] == 1


def test_synthesize_corporate_domain_contact():
    synth = synthesize_corporate_domain_contact("www.ingcapricornio.mx")
    assert synth["domain"] == "ingcapricornio.mx"
    assert synth["synthesized_email"] == "contacto@ingcapricornio.mx"
    assert "rh@ingcapricornio.mx" in synth["alternate_inboxes"]


def test_evaluate_stage2_actionability():
    # Record 1: Target with direct email -> Direct Email
    rec1 = {
        "b2b_status": "target",
        "is_rescued_partial": False,
        "email": "juan@empresa.com",
        "inherited_email": "",
        "website": "",
        "phone": "4720000000",
        "b2b_5d_score": 7.5,
        "estimated_headcount": 50,
    }
    # Record 2: Target without direct email, but has corporate website -> Domain Synthesized
    rec2 = {
        "b2b_status": "target",
        "is_rescued_partial": False,
        "email": "",
        "inherited_email": "",
        "website": "www.proinceautomatizacion.com",
        "phone": "4720000001",
        "b2b_5d_score": 6.8,
        "estimated_headcount": 25,
    }
    # Record 3: Partial with corporate email & high score -> Rescued Tier-1 Corporate
    rec3 = {
        "b2b_status": "partial",
        "is_rescued_partial": False,
        "email": "rcgarcia@hotsson.com",
        "inherited_email": "",
        "website": "www.hotsson.com",
        "phone": "4720000002",
        "b2b_5d_score": 5.2,
        "estimated_headcount": 251,
    }
    # Record 4: Low potential -> Not actionable
    rec4 = {
        "b2b_status": "low_potential",
        "is_rescued_partial": False,
        "email": "tienda@gmail.com",
        "inherited_email": "",
        "website": "",
        "phone": "4720000003",
        "b2b_5d_score": 2.1,
        "estimated_headcount": 2,
    }

    assert evaluate_stage2_actionability(rec1)["is_actionable"] is True
    assert evaluate_stage2_actionability(rec1)["channel"] == "Direct Email (Verified DENUE)"
    
    assert evaluate_stage2_actionability(rec2)["is_actionable"] is True
    assert "Website / Synthesized Domain" in evaluate_stage2_actionability(rec2)["channel"]

    assert evaluate_stage2_actionability(rec3)["is_actionable"] is True
    assert "Tier-1 Corporate Domain" in evaluate_stage2_actionability(rec3)["rescue_reason"]

    assert evaluate_stage2_actionability(rec4)["is_actionable"] is False


def test_strata_midpoint_mapping():
    assert map_employee_strata_midpoint("0 a 5 personas") == 3
    assert map_employee_strata_midpoint("6 a 10 personas") == 8
    assert map_employee_strata_midpoint("11 a 30 personas") == 20
    assert map_employee_strata_midpoint("31 a 50 personas") == 40
    assert map_employee_strata_midpoint("51 a 100 personas") == 75
    assert map_employee_strata_midpoint("101 a 250 personas") == 175
    assert map_employee_strata_midpoint("251 y más personas") == 350
    assert map_employee_strata_midpoint("desconocido") == 0


def test_contact_confidence_hierarchy():
    # Direct DENUE email -> high confidence, low risk, verified
    conf1 = evaluate_contact_confidence("juan@aumovio.com", source="direct_denue")
    assert conf1["email_source"] == "direct_denue"
    assert conf1["email_confidence"] == "high"
    assert conf1["contact_verified"] is True
    assert conf1["contact_risk"] == "low"

    # Scraped visible email -> medium-high confidence
    conf2 = evaluate_contact_confidence("info@ingcapricornio.mx", source="scraped_website")
    assert conf2["email_source"] == "scraped_website"
    assert conf2["email_confidence"] == "medium-high"
    assert conf2["contact_verified"] is True

    # Campus inherited email -> medium confidence, medium risk
    conf3 = evaluate_contact_confidence("contacto@aumovio.com", source="campus_inherited")
    assert conf3["email_source"] == "campus_inherited"
    assert conf3["email_confidence"] == "medium"
    assert conf3["contact_verified"] is False
    assert conf3["contact_risk"] == "medium"

    # Synthesized domain role inbox -> low confidence, high risk, unverified
    conf4 = evaluate_contact_confidence("contacto@proinceautomatizacion.com", source="synthesized_domain")
    assert conf4["email_source"] == "synthesized_domain"
    assert conf4["email_confidence"] == "low"
    assert conf4["contact_verified"] is False
    assert conf4["contact_risk"] == "high"


def test_canonical_account_separation():
    df_sites = pl.DataFrame({
        "clee": ["11037001", "11037002", "11037003"],
        "business_name": ["AUMOVIO MATRIZ", "AUMOVIO PLANTA", "HOTSSON HOTEL"],
        "canonical_entity_id": ["ent_aumovio", "ent_aumovio", "ent_hotsson"],
        "canonical_parent_name": ["AUMOVIO MEXICO", "AUMOVIO MEXICO", "HOTEL HOTSSON"],
        "b2b_status": ["target", "target", "partial"],
        "is_rescued_partial": [False, False, True],
        "is_duplicate": [False, True, False],
        "is_stage2_actionable": [True, True, True],
        "b2b_5d_score": [8.5, 8.2, 5.4],
        "est_employees_midpoint": [350, 75, 175],
        "est_monthly_payroll_mxn": [3500000.0, 750000.0, 1500000.0],
        "effective_contact": ["contacto@aumovio.com", "contacto@aumovio.com", "rcgarcia@hotsson.com"],
        "actionable_channel": ["Direct Email (Verified DENUE)", "Campus Inherited Email (Parent Entity)", "Direct Email (Rescued Tier-1 Corporate)"],
        "email_source": ["direct_denue", "campus_inherited", "direct_denue"],
        "email_confidence": ["high", "medium", "high"],
    })
    canon_master = build_canonical_action_master(df_sites)
    site_list = build_site_action_list(df_sites)

    # Exactly 2 canonical accounts vs 3 site locations
    assert len(canon_master) == 2
    assert len(site_list) == 3
    aumovio_acc = canon_master.filter(pl.col("canonical_entity_id") == "ent_aumovio").to_dicts()[0]
    assert aumovio_acc["site_count"] == 2
    assert aumovio_acc["total_est_employees_midpoint"] == 425
    assert aumovio_acc["consolidated_payroll_mxn"] == 4250000.0


def test_corporate_email_domain_unioning():
    df = pl.DataFrame({
        "clee": ["11037010", "11037011", "11037012", "11037013"],
        "nombre_estab": ["CONSORCIO INDUSTRIAL MEXICANO", "LEAR SILAO", "PAPELERIA EL SOL", "PAPELERIA LA LUNA"],
        "raz_social": ["CONSORCIO INDUSTRIAL MEXICANO DE AUTOPARTES S.A. DE C.V.", "LEAR CORPORATION MEXICO S. DE R.L. DE C.V.", "PAPELERIA EL SOL S.A. DE C.V.", "PAPELERIA LA LUNA S.A. DE C.V."],
        "lat": [20.950, 20.955, 20.940, 20.942],
        "lon": [-101.410, -101.415, -101.400, -101.402],
        "email": ["mmendoza01@lear.com", "mmendoza01@lear.com", "sol@gmail.com", "luna@gmail.com"],
        "b2b_status": ["target", "target", "no_target", "no_target"],
        "b2b_5d_score": [8.0, 7.9, 2.0, 2.1],
        "estimated_headcount": [350, 350, 3, 2],
        "estimated_monthly_payroll_mxn": [10500000.0, 10500000.0, 30000.0, 20000.0],
    })
    clustered = cluster_corporate_entities(df)
    c_lear1 = clustered.filter(pl.col("clee") == "11037010").to_dicts()[0]
    c_lear2 = clustered.filter(pl.col("clee") == "11037011").to_dicts()[0]
    c_sol = clustered.filter(pl.col("clee") == "11037012").to_dicts()[0]
    c_luna = clustered.filter(pl.col("clee") == "11037013").to_dicts()[0]

    # Shared non-free corporate email / domain unifies Lear plants
    assert c_lear1["canonical_entity_id"] == c_lear2["canonical_entity_id"]
    assert c_lear1["campus_site_count"] == 2

    # Free email (gmail.com) must NOT unify unrelated entities
    assert c_sol["canonical_entity_id"] != c_luna["canonical_entity_id"]
    assert c_sol["campus_site_count"] == 1
    assert c_luna["campus_site_count"] == 1


def test_campus_summary_b2b_filtering():
    df_clustered = pl.DataFrame({
        "canonical_entity_id": ["ent_auto", "ent_auto", "ent_walmart", "ent_walmart"],
        "canonical_parent_name": ["AUTO GROUP", "AUTO GROUP", "NUEVA WAL MART", "NUEVA WAL MART"],
        "business_name": ["PLANTA 1", "PLANTA 2", "WALMART SUPER", "WALMART EXPRESS"],
        "campus_site_count": [2, 2, 2, 2],
        "b2b_status": ["target", "target", "no_target", "no_target"],
        "total_est_employees_midpoint": [500, 500, 300, 300],
        "campus_consolidated_payroll_mxn": [15000000.0, 15000000.0, 4500000.0, 4500000.0],
        "inherited_email": ["contact@autogroup.com", "contact@autogroup.com", "info@walmart.com", "info@walmart.com"],
        "inherited_website": ["autogroup.com", "autogroup.com", "walmart.com", "walmart.com"],
    })
    campus_summary = (
        df_clustered.filter((pl.col("campus_site_count") > 1) & (pl.col("b2b_status") != "no_target"))
        .group_by(["canonical_entity_id", "canonical_parent_name"])
        .agg([
            pl.len().alias("sites"),
            pl.col("business_name").str.join(" | ").alias("plantas"),
            pl.col("total_est_employees_midpoint").first().alias("total_headcount"),
            pl.col("campus_consolidated_payroll_mxn").first().alias("nomina_total"),
            pl.col("inherited_email").first().alias("email_campus"),
            pl.col("inherited_website").first().alias("web_campus"),
        ])
        .sort("nomina_total", descending=True)
    )
    entities = campus_summary["canonical_parent_name"].to_list()
    assert "AUTO GROUP" in entities
    assert "NUEVA WAL MART" not in entities



