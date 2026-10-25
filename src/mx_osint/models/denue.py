import time
from mx_osint.config import Settings
settings = Settings()

from datetime import datetime
from pathlib import Path
import random
from typing import Optional
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator
STRATA_ORDER: list[str] = ['0 a 5 personas', '6 a 10 personas', '11 a 30 personas', '31 a 50 personas', '51 a 100 personas', '101 a 250 personas', '251 y más personas']
STRATA_HEADCOUNT_MIDPOINT: dict[str, float] = {'0 a 5 personas': 3.0, '6 a 10 personas': 8.0, '11 a 30 personas': 20.0, '31 a 50 personas': 40.0, '51 a 100 personas': 75.0, '101 a 250 personas': 175.0, '251 y más personas': 350.0}

class DENUEBusiness(BaseModel):
    """Schema for a single DENUE business record after ingest and cleaning."""
    model_config = ConfigDict(str_strip_whitespace=True)
    row_id: str = Field(..., description='Unique row identifier from INEGI')
    denue_id: str = Field(..., description='CLEE unique business ID')
    business_name: str = Field(..., min_length=1, description='Commercial name')
    legal_name: Optional[str] = Field(None, description='Legal entity name (razón social)')
    scian_code: str = Field(..., pattern='^\\d{6}$', description='6-digit SCIAN code')
    scian_description: str = Field(..., min_length=1, description='SCIAN activity description')
    employee_strata: str = Field(..., description='Employee range category')
    lat: float = Field(..., ge=-90, le=90, description='Latitude')
    lon: float = Field(..., ge=-180, le=180, description='Longitude')
    street_type: Optional[str] = None
    street_name: Optional[str] = None
    street_number_ext: Optional[str] = None
    postal_code: Optional[str] = Field(None, pattern='^\\d{5}$')
    state: str = Field(..., min_length=1)
    municipality: str = Field(..., min_length=1)
    locality: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    registration_date: Optional[str] = None

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        return v if v and '@' in v else None

    @field_validator('website')
    @classmethod
    def validate_website(cls, v: Optional[str]) -> Optional[str]:
        return v if v and (v.startswith('http') or v.startswith('www')) else None

class DENUEDataset(BaseModel):
    """Metadata summary schema for a filtered DENUE extract."""
    version: str = Field(default='2026.10')
    state_code: str = Field(..., pattern='^\\d{2}$')
    radius_km: float = Field(..., gt=0)
    target_lat: float
    target_lon: float
    total_records: int = Field(..., ge=0)
    records_with_email: int = Field(..., ge=0)
    records_with_phone: int = Field(..., ge=0)
    records_with_website: int = Field(..., ge=0)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def email_coverage(self) -> float:
        return self.records_with_email / self.total_records * 100 if self.total_records > 0 else 0.0

def generate_synthetic_denue_fixture(output_path: Path=Path('data/raw/synthetic_denue_fallback.parquet'), n_rows: int=500) -> Path:
    """Generates offline fallback DENUE sample centered on target anchor without mutating test fixtures."""
    random.seed(42)
    scian_codes = [('621111', 'Consultorios de medicina general del sector privado'), ('541110', 'Bufetes jurídicos'), ('541211', 'Servicios de contabilidad y auditoría'), ('541330', 'Servicios de ingeniería'), ('541510', 'Servicios de diseño de sistemas de cómputo'), ('336390', 'Fabricación de otras partes para vehículos automotrices'), ('721111', 'Hoteles con otros servicios integrados'), ('522110', 'Banca múltiple')]
    rows = []
    for i in range(n_rows):
        scian_code, scian_desc = random.choice(scian_codes)
        rows.append({'row_id': str(9000000 + i), 'denue_id': f'11037{i:020d}', 'business_name': f'CORPORATIVO SINTETICO {i:04d}', 'legal_name': f'CORPORATIVO SINTETICO {i:04d} SA DE CV', 'scian_code': scian_code, 'scian_description': scian_desc, 'employee_strata': random.choice(STRATA_ORDER[2:]), 'street_type': 'BOULEVARD', 'street_name': 'EMILIANO ZAPATA', 'street_number_ext': str(random.randint(1, 999)), 'postal_code': '36100', 'state': 'GUANAJUATO', 'municipality': 'Leon de los Aldama', 'locality': 'Leon de los Aldama', 'phone': f'472{random.randint(1000000, 9999999)}' if random.random() < 0.6 else None, 'email': f'contacto{i}@empresa{i}.com.mx' if random.random() < 0.5 else None, 'website': f'https://www.empresa{i}.com.mx' if random.random() < 0.35 else None, 'registration_date': '2021-06', 'lat': settings.TARGET_LAT + random.gauss(0, 0.015), 'lon': settings.TARGET_LON + random.gauss(0, 0.015)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path)
    return output_path
