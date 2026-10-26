import json
from pathlib import Path
from mx_osint.models.denue import STRATA_HEADCOUNT_MIDPOINT


def calculate_b2b_emv_record(r: dict) -> dict:
    score_5d = float(r.get('max_b2b_5d_score') or 5.0)
    payroll = float(r.get('consolidated_payroll_mxn') or 500000.0)
    headcount = int(r.get('total_est_employees_midpoint') or 25)
    dist = float(r.get('distance_km') or 1.5)
    chrono = r.get('chronotype_key', 'Office_Standard_8AM')
    conf = r.get('email_confidence', 'none')
    chrono_rates = {'Turno_Ingenieria_630_730AM': 0.28, 'Guardia_Medica_700_830AM': 0.32, 'Ejecutivo_Oficina_800_930AM': 0.35, 'Apertura_Comercial_900_1000AM': 0.18}
    base_rate = chrono_rates.get(chrono, 0.2)
    dist_decay = max(0.4, 1.0 - dist / 8.0)
    capture_rate = round(min(0.4, base_rate * dist_decay), 3)
    white_collar = max(1, round(headcount * 0.4)) if headcount > 50 else max(1, headcount)
    avg_ticket_mxn = float(r.get('avg_ticket_mxn') or 65.0)
    group_order_ticket_mxn = float(r.get('group_order_ticket_mxn') or 174.31)
    working_days = 20
    emv_mxn = round(white_collar * capture_rate * avg_ticket_mxn * working_days, 2)
    p1 = min(50.0, score_5d * 5.0)
    p2 = 20.0 if '630' in chrono or '800' in chrono else 12.0
    p3 = min(15.0, payroll / 5000000.0 * 15.0)
    p4 = 15.0 if conf == 'high' else 10.0 if conf in ['medium-high', 'medium'] else 5.0
    total_score = round(min(100.0, max(15.0, p1 + p2 + p3 + p4)), 1)
    return {'b2b_propensity_score': total_score, 'capture_rate': capture_rate, 'white_collar_headcount': white_collar, 'est_monthly_coffee_rev_mxn': emv_mxn}


class B2BPropensityEnricher:
    """
    Evaluates commercial B2B propensity and Expected Monetary Value (EMV)
    based on SCIAN industry priors, employee strata, and operational chronotype.
    """

    def __init__(self, propensity_path: Path | str | None = None):
        self.propensity_path = Path(propensity_path) if propensity_path else Path('data_contracts/scian_b2b_propensity.json')
        self._weights: dict = {}
        if self.propensity_path.exists():
            try:
                with open(self.propensity_path, 'r', encoding='utf-8') as f:
                    self._weights = json.load(f)
            except Exception:
                self._weights = {}

    def evaluate(
        self,
        scian_code: str | None = None,
        estrato_personal: str | None = None,
        headcount: int | None = None,
        distance_km: float = 1.5,
        chronotype_key: str | None = None,
        **kwargs,
    ) -> dict:
        hc = headcount
        if hc is None and estrato_personal:
            hc = int(STRATA_HEADCOUNT_MIDPOINT.get(estrato_personal, 25))
        if hc is None:
            hc = 25

        chrono = chronotype_key
        if chrono is None and scian_code:
            from mx_osint.enrichment.chronotype_imputer import impute_morning_chronotype_4digit
            chrono = impute_morning_chronotype_4digit(scian_code).get('chronotype_key')
        if chrono is None:
            chrono = 'Office_Standard_8AM'

        rec = {
            'scian_code': scian_code or '',
            'total_est_employees_midpoint': hc,
            'distance_km': distance_km,
            'chronotype_key': chrono,
            **kwargs,
        }
        res = calculate_b2b_emv_record(rec)
        prefix2 = str(scian_code or '')[:2]
        if prefix2 in self._weights:
            res['scian_weight'] = self._weights[prefix2].get('weight', 50.0)
            res['scian_rationale'] = self._weights[prefix2].get('rationale', '')
        return res

