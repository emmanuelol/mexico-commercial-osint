import json
from pathlib import Path


def impute_morning_chronotype_4digit(scian_code: str | None, academic_level: str | None=None) -> dict:
    code = str(scian_code or '').strip()
    prefix2 = code[:2]
    prefix3 = code[:3]
    prefix4 = code[:4]
    if prefix4 in ['3361', '3362', '3363', '5413'] or prefix2 in ['31', '32', '33', '23'] or prefix3 in ['484', '493']:
        return {'chronotype_key': 'Turno_Ingenieria_630_730AM', 'time_window': '06:30 - 08:00 AM', 'demand_driver': 'Cambio de Turno Operativo & Juntas de Piso', 'product_fit': 'Coffee Box Térmico 10-20 pax + Desayuno de Ingeniería', 'badge': "<span style='background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:10px; font-weight:bold; font-size:10px;'>🏭 TURNO PLANTA (6:30-8:00 AM)</span>"}
    elif prefix3 in ['621', '622', '623'] or prefix2 == '62':
        return {'chronotype_key': 'Guardia_Medica_700_830AM', 'time_window': '07:00 - 08:30 AM', 'demand_driver': 'Relevo de Guardia Médica & Visita Clínica', 'product_fit': 'Pack Café Especialidad Baristas + Servicio Express', 'badge': "<span style='background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:10px; font-weight:bold; font-size:10px;'>🏥 GUARDIA MÉDICA (7:00-8:30 AM)</span>"}
    elif prefix4 in ['5415', '5221', '7211'] or prefix2 in ['52', '55'] or academic_level == 'corporate_professional':
        return {'chronotype_key': 'Ejecutivo_Oficina_800_930AM', 'time_window': '08:00 - 09:30 AM', 'demand_driver': 'Llegada Directiva & Salas de Consejo', 'product_fit': 'Barra Ejecutiva & Convenio Corporativo CFDI 4.0', 'badge': "<span style='background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:10px; font-weight:bold; font-size:10px;'>💼 OFICINA / EJECUTIVO (8:00-9:30 AM)</span>"}
    else:
        return {'chronotype_key': 'Apertura_Comercial_900_1000AM', 'time_window': '09:00 - 10:00 AM', 'demand_driver': 'Arranque Comercial & Pausa Matutina', 'product_fit': 'Take-away & Entrega Programada', 'badge': "<span style='background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:10px; font-weight:bold; font-size:10px;'>🏪 COMERCIO (9:00-10:00 AM)</span>"}


class ChronotypeImputer:
    """
    Infers operational chronotypes and B2B demand windows from 2-4 digit SCIAN codes.
    Uses domain Bayesian priors (data_contracts/scian_chronotype_mapping.json) when ground-truth hours are absent.
    """

    def __init__(self, mapping_path: Path | str | None = None):
        self.mapping_path = Path(mapping_path) if mapping_path else Path('data_contracts/scian_chronotype_mapping.json')
        self._mapping: dict = {}
        if self.mapping_path.exists():
            try:
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    self._mapping = json.load(f)
            except Exception:
                self._mapping = {}

    def impute(self, scian_code: str | None, academic_level: str | None = None) -> dict:
        """Imputes morning demand chronotype profile."""
        profile = impute_morning_chronotype_4digit(scian_code, academic_level)
        code = str(scian_code or '').strip()
        prefix2 = code[:2]
        if prefix2 in self._mapping:
            contract_meta = self._mapping[prefix2]
            profile['scian_sector'] = contract_meta.get('sector', '')
            profile['gatekeeper_roles'] = contract_meta.get('gatekeeper_roles', [])
        return profile

