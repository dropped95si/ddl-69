from typing import Any, Dict, Optional, List
from datetime import datetime

from supabase import Client, create_client
from postgrest.exceptions import APIError as PostgrestAPIError

from ddl69.core.settings import Settings



class SupabaseLedger:
    settings: Settings
    client: Client
    """
    Thin wrapper around Supabase tables.
    Uses SERVICE ROLE key for writes (server-side only).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or Settings.from_env()
        if not self.settings.supabase_url or not self.settings.supabase_service_role_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env")

        self.client: Client = create_client(
            self.settings.supabase_url,
            self.settings.supabase_service_role_key,
        )

    def raw(self) -> Client:
        return self.client

    def _exec(self, op_desc: str, fn: callable) -> Any:
        try:
            return fn()
        except PostgrestAPIError as e:
            raise RuntimeError(f"{op_desc} failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"{op_desc} failed: {e}") from e



    # ----------------------------
    # Ingest schema helpers (matches sql/ingest_v1.sql)
    # ----------------------------
    def upsert_instrument(
        self,
        instrument_id: str,
        instrument_type: str = "equity",
        exchange: Optional[str] = None,
        currency: Optional[str] = None,
        meta_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client.table("instruments").upsert(
            {
                "instrument_id": instrument_id,
                "instrument_type": instrument_type,
                "exchange": exchange,
                "currency": currency,
                "meta_json": meta_json or {},
            },
            on_conflict="instrument_id",
        ).execute()


    # -------------------------
    # Bars
    # -------------------------
    def upsert_bars(self, payload: list[Dict[str, Any]]) -> None:
        # matches sql/ingest_v1.sql primary key: instrument_id, provider_id, timeframe, ts
        self._exec(
            "upsert bars",
            lambda: self.client.table("bars").upsert(
                payload,
                on_conflict="instrument_id,provider_id,timeframe,ts",
            ).execute(),
        )
    
    # -------------------------
    # Runs
    # -------------------------
    def create_run(
        self,
        *,
        asof_ts: datetime,
        mode: str,
        config_hash: str,
        code_version: str,
        notes: Optional[str] = None,
        status: str = "created",
    ) -> str:
        payload: Dict[str, Any] = {
            "asof_ts": asof_ts.isoformat(),
            "mode": mode,
            "config_hash": config_hash,
            "code_version": code_version,
            "status": status,
            "notes": notes,
        }
        
        res = self._exec("insert runs", lambda: self.client.table("runs").insert(payload).execute())
        if not getattr(res, "data", None):
            raise RuntimeError("insert runs returned no data")
        return res.data[0]["run_id"]

    # -------------------------
    # Events
    # -------------------------
    def upsert_event(
        self,
        *,
        event_id: str,
        subject_type: str,
        subject_id: str,
        event_type: str,
        asof_ts: datetime,
        horizon_json: Dict[str, Any],
        event_params_json: Dict[str, Any],
        context_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "event_id": event_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "event_type": event_type,
            "asof_ts": asof_ts.isoformat(),
            "horizon_json": horizon_json,
            "event_params_json": event_params_json,
            "context_json": context_json or {},
        }
        self._exec(
            "upsert events",
            lambda: self.client.table("events").upsert(payload, on_conflict="event_id").execute(),
        )

    def upsert_events(self, payload: list[Dict[str, Any]]) -> None:
        if not payload:
            return
        self._exec(
            "upsert events batch",
            lambda: self.client.table("events").upsert(payload, on_conflict="event_id").execute(),
        )

    # -------------------------
    # Forecasts
    # -------------------------
    def insert_expert_forecast(
        self,
        *,
        run_id: str,
        event_id: str,
        expert_name: str,
        expert_version: str,
        probs_json: Dict[str, float],
        confidence: float,
        uncertainty_json: Optional[Dict[str, Any]] = None,
        loss_hint: str = "logloss",
        supports_calibration: bool = True,
        calibration_group: Optional[str] = None,
        features_uri: Optional[str] = None,
        artifact_uris: Optional[List[str]] = None,
        reasons_json: Optional[List[Dict[str, Any]]] = None,
        debug_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "event_id": event_id,
            "expert_name": expert_name,
            "expert_version": expert_version,
            "probs_json": probs_json,
            "confidence": confidence,
            "uncertainty_json": uncertainty_json or {},
            "loss_hint": loss_hint,
            "supports_calibration": supports_calibration,
            "calibration_group": calibration_group,
            "features_uri": features_uri,
            "artifact_uris": artifact_uris or [],
            "reasons_json": reasons_json or [],
            "debug_json": debug_json or {},
        }
        self._exec("insert expert_forecasts", lambda: self.client.table("expert_forecasts").insert(payload).execute())

    def upsert_expert_forecasts(self, payload: list[Dict[str, Any]]) -> None:
        if not payload:
            return
        self._exec(
            "upsert expert_forecasts batch",
            lambda: self.client.table("expert_forecasts").upsert(
                payload, on_conflict="run_id,event_id,expert_name"
            ).execute(),
        )

    def insert_ensemble_forecast(
        self,
        *,
        run_id: str,
        event_id: str,
        method: str,
        probs_json: Dict[str, float],
        confidence: float,
        uncertainty_json: Optional[Dict[str, Any]] = None,
        weights_json: Optional[Dict[str, float]] = None,
        explain_json: Optional[Dict[str, Any]] = None,
        artifact_uris: Optional[List[str]] = None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "event_id": event_id,
            "method": method,
            "probs_json": probs_json,
            "confidence": confidence,
            "uncertainty_json": uncertainty_json or {},
            "weights_json": weights_json or {},
            "explain_json": explain_json or {},
            "artifact_uris": artifact_uris or [],
        }
        self._exec("insert ensemble_forecasts", lambda: self.client.table("ensemble_forecasts").insert(payload).execute())

    def upsert_ensemble_forecasts(self, payload: list[Dict[str, Any]]) -> None:
        if not payload:
            return
        self._exec(
            "upsert ensemble_forecasts batch",
            lambda: self.client.table("ensemble_forecasts").upsert(
                payload, on_conflict="run_id,event_id,method"
            ).execute(),
        )

    # -------------------------
    # Artifacts
    # -------------------------
    def insert_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        uri: str,
        sha256: Optional[str] = None,
        row_count: Optional[int] = None,
        meta_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "kind": kind,
            "uri": uri,
            "sha256": sha256,
            "row_count": row_count,
            "meta_json": meta_json or {},
        }
        self._exec("insert artifacts", lambda: self.client.table("artifacts").insert(payload).execute())

    # -------------------------
    # Storage
    # -------------------------
    def upload_storage(
        self,
        *,
        bucket: str,
        local_path: str,
        dest_path: str,
        upsert: bool = True,
    ) -> str:
        with open(local_path, "rb") as f:
            self._exec(
                "upload storage",
                lambda: self.client.storage.from_(bucket).upload(
                    dest_path,
                    f,
                    file_options={"upsert": upsert},
                ),
            )
        return f"supabase://{bucket}/{dest_path}"
