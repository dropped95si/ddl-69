from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    supabase_url: str = os.getenv("SUPABASE_URL", "").strip()
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    artifact_root: str = os.getenv("ARTIFACT_ROOT", "./artifacts").strip()
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns").strip()

    def validate(self) -> None:
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env")

SETTINGS = Settings()
