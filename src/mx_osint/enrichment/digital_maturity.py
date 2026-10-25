import time
from loguru import logger

def classify_digital_maturity_record(row: dict) -> dict:
    has_em = bool(row.get('effective_contact') and str(row.get('effective_contact')).strip() not in ['', '-', 'None'])
    conf = (row.get('email_confidence') or 'none').lower()
    has_web = bool(row.get('website') and str(row.get('website')).strip() not in ['', '-', 'None'])
    has_ph = bool(row.get('phone') and str(row.get('phone')).strip() not in ['', '-', 'None'])
    score = 0
    if has_em and conf in ['high', 'medium-high']:
        score += 2
    elif has_em:
        score += 1
    if has_web:
        score += 1
    return {'has_verified_email': bool(has_em and conf in ['high', 'medium-high']), 'has_website': has_web, 'has_phone': has_ph, 'digital_maturity_score': min(3, score), 'digital_gap_flag': not has_em or conf in ['low', 'none']}

def discover_channel_ddg(entity_name: str, state: str='Guanajuato', max_results: int=2) -> dict:
    cache_key = f'{entity_name}::{state}'
    if cache_key in ddg_cache:
        return ddg_cache[cache_key]
    query = f'"{entity_name}" "{state}" contacto OR telefono OR recursos humanos'
    out = {'status': 'searched', 'top_url': None, 'snippet': None}
    try:
        from duckduckgo_search import DDGS
        time.sleep(0.5)
        with DDGS() as ddgs:
            res = list(ddgs.text(query, max_results=max_results))
            if res:
                out = {'status': 'found', 'top_url': res[0].get('href'), 'snippet': res[0].get('body', '')[:120]}
    except Exception as e:
        out = {'status': 'skipped_or_offline', 'top_url': f'https://www.google.com/search?q={entity_name}+Guanajuato', 'snippet': 'Búsqueda web directa'}
    ddg_cache[cache_key] = out
    return out
