from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from urllib.parse import quote, urljoin
import polars as pl
import requests
from IPython.display import HTML, display
EMAIL_REGEX = re.compile('[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}')

def classify_b2b_email_role(email: str | None) -> str:
    if not email or not isinstance(email, str) or '@' not in email:
        return 'none'
    clean_em = email.strip().lower()
    free_domains = ['gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com', 'live.com', 'prodigy.net.mx']
    if any((clean_em.endswith(f'@{d}') for d in free_domains)):
        return 'free_personal'
    user_part = clean_em.split('@')[0]
    decision_makers = ['director', 'gerente', 'rh', 'recursoshumanos', 'purchasing', 'compras', 'talento', 'ceo', 'cfo', 'planta']
    generic_roles = ['contacto', 'info', 'recepcion', 'atencion', 'ventas', 'servicio', 'facturacion', 'admin']
    if any((dm in user_part for dm in decision_makers)):
        return 'decision_maker_role'
    if any((gen in user_part for gen in generic_roles)):
        return 'generic_role_inbox'
    return 'corporate_direct'

def synthesize_corporate_domain_contact(website: str | None) -> dict:
    if not website or not isinstance(website, str):
        return {'domain': '', 'synthesized_email': '', 'alternate_inboxes': []}
    clean_w = website.strip().lower()
    for pfx in ['https://', 'http://', 'www.']:
        if clean_w.startswith(pfx):
            clean_w = clean_w[len(pfx):]
    clean_w = clean_w.split('/')[0].split('?')[0].strip()
    non_corp = ['facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com', 'google.com', 'maps.google.com', 'gob.mx', 'wixsite.com']
    if not clean_w or '.' not in clean_w or any((clean_w.endswith(fd) for fd in non_corp)):
        return {'domain': '', 'synthesized_email': '', 'alternate_inboxes': []}
    return {'domain': clean_w, 'synthesized_email': f'contacto@{clean_w}', 'alternate_inboxes': [f'rh@{clean_w}', f'compras@{clean_w}', f'ventas@{clean_w}']}

def scrape_corporate_contact_emails(websites: list[str], max_workers: int=8, timeout: float=2.5) -> dict[str, str]:
    scraped = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MexicoCommercialOSINT/1.0'}
    clean_targets = list(set((w.strip() for w in websites if w and isinstance(w, str) and ('.' in w))))

    def scrape_url(w: str):
        url = w if w.lower().startswith('http') else f'https://{w}'
        for target in [url, urljoin(url, '/contacto'), urljoin(url, '/contact')]:
            try:
                resp = requests.get(target, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    found = EMAIL_REGEX.findall(resp.text)
                    valid = [e for e in found if not any((e.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.js', '.css'])) and (not any((e.lower().startswith(p) for p in ['bootstrap@', 'jquery@', 'user@']))) and (len(e) <= 50)]
                    if valid:
                        return (w, valid[0])
            except Exception:
                pass
        return (w, '')
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(scrape_url, w) for w in clean_targets]
        for f in as_completed(futures):
            w_res, em_res = f.result()
            if em_res:
                scraped[w_res] = em_res
    return scraped

def evaluate_stage2_actionability(rec: dict, scraped_map: dict | None=None) -> dict:
    b2b_status = rec.get('b2b_status', '')
    is_rescued = rec.get('is_rescued_partial', False)
    email = (rec.get('email') or '').strip()
    inherited_em = (rec.get('inherited_email') or '').strip()
    website = (rec.get('website') or '').strip()
    phone = (rec.get('phone') or '').strip()
    score = float(rec.get('b2b_5d_score') or 0.0)
    headcount = int(rec.get('est_employees_midpoint') or 0)
    synth = synthesize_corporate_domain_contact(website)
    if (b2b_status == 'target' or is_rescued) and email:
        return {'is_stage2_actionable': True, 'actionable_channel': 'Direct Email (Verified DENUE)', 'rescue_reason': 'Stage-01 Verified Direct Email', 'effective_contact': email, 'email_source': 'direct_denue', 'email_confidence': 'high', 'contact_verified': True, 'contact_risk': 'low', 'actionable_tier': 'Tier-1 Direct Email'}
    if (b2b_status == 'target' or is_rescued) and inherited_em:
        return {'is_stage2_actionable': True, 'actionable_channel': 'Campus Inherited Email (Parent Entity)', 'rescue_reason': 'Campus Contact Inheritance', 'effective_contact': inherited_em, 'email_source': 'campus_inherited', 'email_confidence': 'medium', 'contact_verified': False, 'contact_risk': 'medium', 'actionable_tier': 'Tier-1 Campus Inherited'}
    scraped_em = (scraped_map or {}).get(website, '')
    if (b2b_status == 'target' or is_rescued) and scraped_em:
        return {'is_stage2_actionable': True, 'actionable_channel': 'Direct Email (Scraped Corporate Website)', 'rescue_reason': 'Live Corporate Website Email Scraping', 'effective_contact': scraped_em, 'email_source': 'scraped_website', 'email_confidence': 'medium-high', 'contact_verified': True, 'contact_risk': 'low', 'actionable_tier': 'Tier-1 Direct Email'}
    if (b2b_status == 'target' or is_rescued) and synth['synthesized_email']:
        return {'is_stage2_actionable': True, 'actionable_channel': 'Website / Synthesized Domain Email', 'rescue_reason': 'Corporate Web Domain Contact Synthesis', 'effective_contact': synth['synthesized_email'], 'email_source': 'synthesized_domain', 'email_confidence': 'low', 'contact_verified': False, 'contact_risk': 'high', 'actionable_tier': 'Tier-2 Domain Synthesized'}
    if b2b_status == 'partial' and score >= STAGE2_CONFIG['partial_upgrade_score_threshold'] and email:
        return {'is_stage2_actionable': True, 'actionable_channel': 'Direct Email (Rescued Tier-1 Corporate)', 'rescue_reason': 'Tier-1 Corporate Domain Partial Upgrade', 'effective_contact': email, 'email_source': 'direct_denue', 'email_confidence': 'high', 'contact_verified': True, 'contact_risk': 'low', 'actionable_tier': 'Tier-1 Rescued Partial'}
    elif b2b_status == 'partial' and score >= STAGE2_CONFIG['partial_upgrade_score_threshold'] and synth['synthesized_email'] and (headcount >= STAGE2_CONFIG['min_headcount_corporate_partial']):
        return {'is_stage2_actionable': True, 'actionable_channel': 'Website / Synthesized Domain Email', 'rescue_reason': 'Tier-1 Headcount & Domain Partial Upgrade', 'effective_contact': synth['synthesized_email'], 'email_source': 'synthesized_domain', 'email_confidence': 'low', 'contact_verified': False, 'contact_risk': 'high', 'actionable_tier': 'Tier-2 Rescued Partial'}
    if (b2b_status == 'target' or is_rescued) and phone:
        return {'is_stage2_actionable': True, 'actionable_channel': 'Direct Phone + LinkedIn ABM', 'rescue_reason': 'Multi-Channel Phone & Executive Search Rescue', 'effective_contact': f'Tel: {phone}', 'email_source': 'none', 'email_confidence': 'none', 'contact_verified': True, 'contact_risk': 'medium', 'actionable_tier': 'Tier-3 Phone / ABM'}
    if b2b_status == 'target' or is_rescued:
        return {'is_stage2_actionable': True, 'actionable_channel': 'Field Sales + LinkedIn ABM', 'rescue_reason': 'Storefront Physical Visit & Executive Search Rescue', 'effective_contact': 'Visita Presencial / Ruta', 'email_source': 'none', 'email_confidence': 'none', 'contact_verified': False, 'contact_risk': 'high', 'actionable_tier': 'Tier-4 Field Sales / ABM'}
    return {'is_stage2_actionable': False, 'actionable_channel': 'Non-Actionable (Low Priority)', 'rescue_reason': 'Does not meet commercial qualification criteria', 'effective_contact': '-', 'email_source': 'none', 'email_confidence': 'none', 'contact_verified': False, 'contact_risk': 'low', 'actionable_tier': 'Inactive'}
