import hashlib
from pathlib import Path
import re
import unicodedata
from collections import defaultdict
import polars as pl
from rapidfuzz import fuzz
from IPython.display import HTML, display
LEGAL_SUFFIX_REGEX = re.compile('\\b(S\\.?\\s*A\\.?\\s*P\\.?\\s*I\\.?\\s*(?:DE\\s*C\\.?\\s*V\\.?)?|S\\.?\\s*A\\.?\\s*B\\.?\\s*(?:DE\\s*C\\.?\\s*V\\.?)?|S\\.?\\s*A\\.?\\s*DE\\s*C\\.?\\s*V\\.?|S\\.?\\s*DE\\s*R\\.?\\s*L\\.?\\s*(?:DE\\s*C\\.?\\s*V\\.?)?|S\\.?\\s*C\\.?\\s*DE\\s*R\\.?\\s*L\\.?|S\\.?\\s*C\\.?|A\\.?\\s*C\\.?|S\\.?\\s*A\\.?|SAB\\s*DE\\s*CV)\\b', re.IGNORECASE)
BRANCH_QUALIFIER_REGEX = re.compile('\\b(PLANTA|MATRIZ|SUCURSAL|CORPORATIVO|CAMPUS|BODEGA|NORTE|SUR|LEON|GUANAJUATO|BAJIO|MEXICO)\\b', re.IGNORECASE)
PUNCTUATION_REGEX = re.compile('[,.\\-\'\\"]')
SPANISH_NAME_STOPWORDS = {'DE', 'DEL', 'LA', 'EL', 'LOS', 'LAS', 'Y', 'EN', 'PARA', 'CON'}

def normalize_mexican_legal_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ''
    text = unicodedata.normalize('NFKD', name)
    text = ''.join((c for c in text if not unicodedata.combining(c))).upper()
    text = LEGAL_SUFFIX_REGEX.sub(' ', text)
    text = PUNCTUATION_REGEX.sub(' ', text)
    return ' '.join(text.split()).strip()

def normalize_mexican_business_name(name: str) -> str:
    """
    Normalizes Mexican business trade names for deterministic cross-source matching:
    removes diacritics, strips statutory legal suffixes, punctuation, and common Spanish stopwords,
    returning a clean lowercased token sequence.
    """
    if not name or not isinstance(name, str):
        return ''
    cleaned = normalize_mexican_legal_name(name)
    tokens = [t.lower() for t in cleaned.split() if t.upper() not in SPANISH_NAME_STOPWORDS]
    return ' '.join(tokens)

def strip_corporate_branch_qualifiers(name: str) -> str:
    cleaned = normalize_mexican_legal_name(name)
    cleaned = BRANCH_QUALIFIER_REGEX.sub(' ', cleaned)
    cleaned = PUNCTUATION_REGEX.sub(' ', cleaned)
    return ' '.join(cleaned.split()).strip()

def classify_workforce_academic_level(wage_mxn: float | None, degree_share: float | None) -> str:
    wage = float(wage_mxn or 0.0)
    share = float(degree_share or 0.0)
    if share >= 0.6 and wage >= 25000:
        return 'high_automation_engineering'
    elif share >= 0.35 and wage >= 18000:
        return 'corporate_professional'
    elif share >= 0.2 or wage >= 15000:
        return 'specialized_technical'
    else:
        return 'traditional_labor'

def cluster_corporate_entities(df: pl.DataFrame, fuzzy_threshold: float=0.85) -> pl.DataFrame:
    name_col = 'business_name' if 'business_name' in df.columns else 'nombre_estab'
    legal_col = 'legal_name' if 'legal_name' in df.columns else 'raz_social'
    norm_commercial = [normalize_mexican_legal_name(str(n or '')) for n in df[name_col].to_list()]
    parent_commercial = [strip_corporate_branch_qualifiers(str(n or '')) for n in df[name_col].to_list()]
    norm_legal = [normalize_mexican_legal_name(str(l or '')) if l else '' for l in df[legal_col].to_list()]
    parent_legal = [strip_corporate_branch_qualifiers(str(l or '')) if l else '' for l in df[legal_col].to_list()]
    df_prep = df.with_columns([pl.Series('norm_name', norm_commercial), pl.Series('parent_name', parent_commercial), pl.Series('norm_legal', norm_legal), pl.Series('parent_legal', parent_legal)])
    GENERIC_STOPWORDS = {'SERVICIOS', 'COMERCIAL', 'GRUPO', 'DISTRIBUIDORA', 'LOGISTICA', 'CENTRO', 'DESPACHO', 'TIENDA', 'ABARROTES', 'TALLER', 'FARMACIA', 'CORPORATIVO', 'CONSULTORIA', 'SOLUCIONES', 'TRANSPORTES', 'INMOBILIARIA', 'DESARROLLO', 'TECNOLOGIA'}
    records = df_prep.to_dicts()
    n = len(records)
    parent_map = list(range(n))

    def find(i):
        if parent_map[i] == i:
            return i
        parent_map[i] = find(parent_map[i])
        return parent_map[i]

    def union(i, j):
        ri, rj = (find(i), find(j))
        if ri != rj:
            parent_map[rj] = ri
    FREE_DOMAINS = {'gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com', 'live.com', 'live.com.mx', 'prodigy.net.mx', 'gob.mx'}
    for i in range(n):
        ri = records[i]
        legi = (ri.get('parent_legal') or ri.get(legal_col) or '').strip().upper()
        pcommi = (ri.get('parent_name') or ri.get(name_col) or '').strip().upper()
        emi = (ri.get('email') or '').strip().lower()
        domi = emi.split('@')[1] if '@' in emi and emi.split('@')[1] not in FREE_DOMAINS else ''
        for j in range(i + 1, n):
            rj = records[j]
            legj = (rj.get('parent_legal') or rj.get(legal_col) or '').strip().upper()
            pcommj = (rj.get('parent_name') or rj.get(name_col) or '').strip().upper()
            emj = (rj.get('email') or '').strip().lower()
            domj = emj.split('@')[1] if '@' in emj and emj.split('@')[1] not in FREE_DOMAINS else ''
            match = False
            if legi and legj and (legi == legj) and (len(legi) >= 5) and (legi not in GENERIC_STOPWORDS):
                match = True
            elif pcommi and pcommj and (pcommi == pcommj) and (len(pcommi) >= 5) and (pcommi not in GENERIC_STOPWORDS):
                match = True
            elif emi and emj and (emi == emj):
                match = True
            elif domi and domj and (domi == domj) and (not any((k in domi for k in ['edu.mx', 'org.mx']))):
                match = True
            elif legi and legj and (len(legi) >= 8) and (len(legj) >= 8) and (legi not in GENERIC_STOPWORDS) and (legj not in GENERIC_STOPWORDS):
                if fuzz.token_sort_ratio(legi, legj) >= 92:
                    match = True
            if match:
                union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    canonical_ids = [None] * n
    canonical_names = [None] * n
    is_duplicate = [False] * n
    inherited_emails = [None] * n
    inherited_websites = [None] * n
    inherited_phones = [None] * n
    contact_inheritance_scores = ['none'] * n
    for root, indices in clusters.items():

        def priority(idx):
            r = records[idx]
            name = str(r.get(name_col, '')).upper()
            is_mat = 1 if 'MATRIZ' in name or 'CORPORATIVO' in name else 0
            has_em = 1 if r.get('email') else 0
            score = float(r.get('b2b_5d_score') or 0.0)
            return (is_mat, has_em, score)
        sorted_idx = sorted(indices, key=priority, reverse=True)
        prim_idx = sorted_idx[0]
        prim_rec = records[prim_idx]
        leg_clean = normalize_mexican_legal_name(prim_rec.get(legal_col) or '')
        canon_name = leg_clean if leg_clean and len(leg_clean) >= 5 and (leg_clean not in GENERIC_STOPWORDS) else prim_rec.get('parent_name') or prim_rec.get(name_col)
        canon_id = f"ent_{hashlib.sha1(canon_name.encode('utf-8')).hexdigest()[:12]}"
        best_em = next((str(records[idx]['email']).strip() for idx in sorted_idx if records[idx].get('email')), '')
        best_web = next((str(records[idx]['website']).strip() for idx in sorted_idx if records[idx].get('website')), '')
        best_ph = next((str(records[idx]['phone']).strip() for idx in sorted_idx if records[idx].get('phone')), '')
        for rank, idx in enumerate(sorted_idx):
            canonical_ids[idx] = canon_id
            canonical_names[idx] = canon_name
            is_duplicate[idx] = rank > 0
            curr_em = str(records[idx].get('email') or '').strip()
            curr_web = str(records[idx].get('website') or '').strip()
            curr_ph = str(records[idx].get('phone') or '').strip()
            inherited_emails[idx] = best_em if not curr_em and best_em else curr_em
            inherited_websites[idx] = best_web if not curr_web and best_web else curr_web
            inherited_phones[idx] = best_ph if not curr_ph and best_ph else curr_ph
            if not curr_em and best_em:
                contact_inheritance_scores[idx] = 'high' if 'MATRIZ' in str(prim_rec.get(name_col, '')).upper() else 'medium'
            elif curr_em:
                contact_inheritance_scores[idx] = 'direct'
    df_out = df_prep.with_columns([pl.Series('canonical_entity_id', canonical_ids), pl.Series('canonical_parent_name', canonical_names), pl.Series('is_duplicate', is_duplicate), pl.Series('inherited_email', inherited_emails), pl.Series('inherited_website', inherited_websites), pl.Series('inherited_phone', inherited_phones), pl.Series('contact_inheritance_score', contact_inheritance_scores)])
    counts = df_out.group_by('canonical_entity_id').agg([pl.len().alias('campus_site_count'), pl.col('est_employees_midpoint').sum().alias('total_est_employees_midpoint'), pl.col('est_monthly_payroll_mxn').sum().alias('campus_consolidated_payroll_mxn')])
    return df_out.join(counts, on='canonical_entity_id', how='left')
