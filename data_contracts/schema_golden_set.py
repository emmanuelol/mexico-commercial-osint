from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class BusinessEntity(BaseModel):
    """Pydantic model representing a Mexican business entity record from DENUE."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Internal or DENUE unique identifier")
    clee: Optional[str] = Field(default=None, description="Clave Estadistica Empresarial")
    nombre_estab: str = Field(description="Trade name (nombre de establecimiento)")
    raz_social: Optional[str] = Field(default=None, description="Legal business name")
    codigo_act: str = Field(description="SCIAN activity code (6 digits)")
    nombre_act: Optional[str] = Field(default=None, description="SCIAN activity description")
    per_ocu: str = Field(description="Employment strata description")
    telefono: Optional[str] = Field(default=None, description="Phone number")
    correoelec: Optional[str] = Field(default=None, description="Email address")
    www: Optional[str] = Field(default=None, description="Website URL")
    lat: float = Field(description="Latitude in WGS84")
    lon: float = Field(description="Longitude in WGS84")
    ent: Optional[str] = Field(default=None, description="State code (2 digits)")
    mun: Optional[str] = Field(default=None, description="Municipality code (3 digits)")
    fecha_alta: Optional[str] = Field(default=None, description="Date added in YYYY-MM format")


class ResolvedMatch(BaseModel):
    """Pairwise entity resolution match record."""
    model_config = ConfigDict(extra="ignore")

    entity_id_a: str
    entity_id_b: str
    match_score: float = Field(ge=0.0, le=1.0)
    is_match: bool
    canonical_entity_id: str


class ProspectScore(BaseModel):
    """Scored prospect for B2B commercial outreach."""
    model_config = ConfigDict(extra="ignore")

    entity_id: str
    nombre_estab: str
    b2b_score: float = Field(ge=0.0, le=100.0)
    channel: str
    digital_maturity_score: int = Field(ge=0, le=3)
    chronotype: str
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
