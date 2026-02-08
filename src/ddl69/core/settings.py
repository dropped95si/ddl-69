from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    supabase_url: str = (
        os.getenv("SUPABASE_URL", "").strip()
        or os.getenv("SUPABASE_URl", "").strip()
    )
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    artifact_root: str = os.getenv("ARTIFACT_ROOT", "./artifacts").strip()
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns").strip()
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "").strip()
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "").strip()
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "").strip()
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "").strip()
    supabase_storage_bucket: str = os.getenv("SUPABASE_STORAGE_BUCKET", "artifacts").strip()
    watchlist: str = os.getenv("WATCHLIST", "").strip()
    massive_access_key: str = (
        os.getenv("MASSIVE_ACCESS_KEY", "").strip()
        or os.getenv("MASSIVE_ACCESS_KEY_ID", "").strip()
    )
    massive_secret_key: str = (
        os.getenv("MASSIVE_SECRET_KEY", "").strip()
        or os.getenv("MASSIVE_SECRET_ACCESS_KEY", "").strip()
    )
    massive_s3_endpoint: str = os.getenv("MASSIVE_S3_ENDPOINT", "").strip()
    massive_s3_bucket: str = os.getenv("MASSIVE_S3_BUCKET", "").strip()
    massive_region: str = os.getenv("MASSIVE_REGION", "us-east-1").strip()

    def supabase_url_normalized(self) -> str:
        url = self.supabase_url
        if url and not url.endswith("/"):
            url += "/"
        return url

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()

    def validate(self) -> None:
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env")

SETTINGS = Settings()
