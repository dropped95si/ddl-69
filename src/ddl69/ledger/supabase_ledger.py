from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

from supabase import create_client, Client

from ddl69.core.settings import Settings

class SupabaseLedger:
    """Thin wrapper around Supabase tables defined in sql/ledger_v1.sql + v2 patch.

    Uses service role key for writes.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()
        if not self.settings.supabase_url or not self.settings.supabase_service_role_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env")
        self.client: Client = create_client(self.settings.supabase_url, self.settings.supabase_service_role_key)

    def insert_run(self, *, asof_ts: datetime, mode: str, config_hash: str, code_version: str, status: str = "created") -> str:
        payload = {
            "asof_ts": asof_ts.isoformat(),
            "mode": mode,
            "config_hash": config_hash,
            "code_version": code_version,
            "status": status,
        }
        res = self.client.table("runs").insert(payload).execute()
        return res.data[0]["run_id"]

    def upsert_event(self, *, event_id: str, subject_type: str, subject_id: str, event_type: str,
                    asof_ts: datetime, horizon_json: Dict[str, Any], event_params_json: Dict[str, Any], context_json: Optional[Dict[str, Any]] = None) -> None:
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
        # Supabase upsert needs a conflict target; event_id is PK
        self.client.table("events").upsert(payload, on_conflict="event_id").execute()

    def insert_expert_forecast(self, *, run_id: str, event_id: str, expert_name: str, expert_version: str,
                              probs_json: Dict[str, float], confidence: float,
                              uncertainty_json: Optional[Dict[str, Any]] = None,
                              loss_hint: str = "logloss",
                              supports_calibration: bool = True,
                              calibration_group: Optional[str] = None,
                              features_uri: Optional[str] = None,
                              artifact_uris: Optional[list[str]] = None,
                              reasons_json: Optional[list[Dict[str, Any]]] = None,
                              debug_json: Optional[Dict[str, Any]] = None) -> None:
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
        self.client.table("expert_forecasts").insert(payload).execute()

    def insert_ensemble_forecast(self, *, run_id: str, event_id: str, method: str,
                                probs_json: Dict[str, float], confidence: float,
                                uncertainty_json: Optional[Dict[str, Any]] = None,
                                weights_json: Optional[Dict[str, float]] = None,
                                explain_json: Optional[Dict[str, Any]] = None,
                                artifact_uris: Optional[list[str]] = None) -> None:
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
        self.client.table("ensemble_forecasts").insert(payload).execute()

    def insert_outcome(self, *, event_id: str, realized_ts: datetime, realized_label: str, realized_meta_json: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "event_id": event_id,
            "realized_ts": realized_ts.isoformat(),
            "realized_label": realized_label,
            "realized_meta_json": realized_meta_json or {},
        }
        self.client.table("event_outcomes").upsert(payload, on_conflict="event_id").execute()

    def insert_weight_update(self, *, asof_ts: datetime, context_key: str, method: str,
                             weights_before: Dict[str, float], weights_after: Dict[str, float],
                             losses: Optional[Dict[str, float]] = None,
                             event_id: Optional[str] = None,
                             run_id: Optional[str] = None) -> None:
        payload = {
            "asof_ts": asof_ts.isoformat(),
            "context_key": context_key,
            "method": method,
            "weights_before_json": weights_before,
            "weights_after_json": weights_after,
            "losses_json": losses or {},
            "event_id": event_id,
            "run_id": run_id,
        }
        self.client.table("weight_updates").insert(payload).execute()
