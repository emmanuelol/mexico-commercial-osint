from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Configuration settings loaded from environment or .env file."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    INEGI_STATE_CODE: str = Field(default="11", description="INEGI State Code (01-32). Default Guanajuato")
    TARGET_LAT: float = Field(default=21.1221, description="Target search latitude")
    TARGET_LON: float = Field(default=-101.6826, description="Target search longitude")
    RADIUS_KM: float = Field(default=3.0, description="Search radius in kilometers")
    MIN_EMPLOYEES_STRATA: str = Field(
        default="11 a 30 personas",
        description="Minimum employee strata threshold"
    )
    DDG_SLEEP_SEC: float = Field(
        default=2.0,
        description="Sleep delay in seconds between DuckDuckGo queries"
    )
    CACHE_DIR: Path = Field(
        default=Path("data/raw"),
        description="Directory for caching downloaded datasets"
    )
    PROCESSED_DIR: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed output datasets"
    )
    ANCHOR_LABEL: str = Field(
        default="Anchor Store — Bajío Hub",
        description="Target anchor label"
    )
