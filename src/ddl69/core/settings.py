from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SUPABASE_URL = ""
DEFAULT_SUPABASE_SERVICE_ROLE_KEY = ""
DEFAULT_POLYGON_API_KEY = ""
DEFAULT_ALPACA_API_KEY = ""
DEFAULT_ALPACA_SECRET_KEY = ""
DEFAULT_ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"


@dataclass(frozen=True)
class Settings:
    supabase_url: str = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL).strip()
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", DEFAULT_SUPABASE_SERVICE_ROLE_KEY).strip()
    artifact_root: str = os.getenv("ARTIFACT_ROOT", "./artifacts").strip()
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns").strip()
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", DEFAULT_POLYGON_API_KEY).strip()
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", DEFAULT_ALPACA_API_KEY).strip()
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", DEFAULT_ALPACA_SECRET_KEY).strip()
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", DEFAULT_ALPACA_BASE_URL).strip()
    supabase_storage_bucket: str = os.getenv("SUPABASE_STORAGE_BUCKET", "artifacts").strip()
    watchlist: str = os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,TSLA,SPY,QQQ").strip()
    massive_s3_endpoint: str = os.getenv("MASSIVE_S3_ENDPOINT", "https://files.massive.com").strip()
    massive_s3_bucket: str = os.getenv("MASSIVE_S3_BUCKET", "").strip()
    massive_access_key: str = os.getenv("MASSIVE_ACCESS_KEY", "").strip()
    massive_secret_key: str = os.getenv("MASSIVE_SECRET_KEY", "").strip()
    massive_region: str = os.getenv("MASSIVE_REGION", "us-east-1").strip() or "us-east-1"

    def validate(self) -> None:
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env")

    @classmethod
    def from_env(cls) -> Settings:
        """Create Settings instance from environment variables."""
        return cls()

SETTINGS = Settings()
