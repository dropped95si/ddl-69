-- Ledger V1 (base tables) — run before ledger_v2_patch.sql
create extension if not exists pgcrypto;

-- runs
create table if not exists public.runs (
  run_id        uuid primary key default gen_random_uuid(),
  asof_ts       timestamptz not null,
  mode          text not null check (mode in ('lean','qlib')),
  config_hash   text not null,
  code_version  text not null,
  status        text not null default 'created'
                check (status in ('created','running','success','failed','canceled')),
  notes         text,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists runs_asof_ts_idx on public.runs(asof_ts desc);
create index if not exists runs_mode_asof_idx on public.runs(mode, asof_ts desc);

-- events
create table if not exists public.events (
  event_id          text primary key,
  subject_type      text not null,
  subject_id        text not null,
  event_type        text not null check (event_type in ('state_event','barrier_event')),
  asof_ts           timestamptz not null,
  horizon_json      jsonb not null,
  event_params_json jsonb not null,
  context_json      jsonb,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create index if not exists events_subject_asof_idx on public.events(subject_id, asof_ts desc);
create index if not exists events_type_asof_idx on public.events(event_type, asof_ts desc);

-- artifacts
create table if not exists public.artifacts (
  artifact_id  uuid primary key default gen_random_uuid(),
  run_id       uuid not null references public.runs(run_id) on delete cascade,
  kind         text not null check (kind in ('features','expert_preds','ensemble_preds','folds','raw','labels','other')),
  uri          text not null,
  sha256       text,
  row_count    bigint,
  meta_json    jsonb,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  unique (run_id, kind, uri)
);

create index if not exists artifacts_run_kind_idx on public.artifacts(run_id, kind);

-- expert_forecasts
create table if not exists public.expert_forecasts (
  id                  uuid primary key default gen_random_uuid(),
  run_id              uuid not null references public.runs(run_id) on delete cascade,
  event_id            text not null references public.events(event_id) on delete cascade,
  expert_name         text not null,
  expert_version      text not null,
  probs_json          jsonb not null,
  confidence          double precision not null check (confidence >= 0 and confidence <= 1),
  uncertainty_json    jsonb,
  loss_hint           text not null default 'logloss' check (loss_hint in ('logloss','brier')),
  supports_calibration boolean not null default true,
  calibration_group   text,
  features_uri        text,
  artifact_uris       jsonb,
  reasons_json        jsonb,
  debug_json          jsonb,
  created_at          timestamptz not null default now(),
  unique (run_id, event_id, expert_name)
);

create index if not exists expert_forecasts_event_idx on public.expert_forecasts(event_id);
create index if not exists expert_forecasts_expert_time_idx on public.expert_forecasts(expert_name, created_at desc);
create index if not exists expert_forecasts_run_idx on public.expert_forecasts(run_id);

-- ensemble_forecasts
create table if not exists public.ensemble_forecasts (
  id               uuid primary key default gen_random_uuid(),
  run_id           uuid not null references public.runs(run_id) on delete cascade,
  event_id         text not null references public.events(event_id) on delete cascade,
  method           text not null check (method in ('hedge','bayes','vw','stacked','blended')),
  probs_json       jsonb not null,
  confidence       double precision not null check (confidence >= 0 and confidence <= 1),
  uncertainty_json jsonb,
  weights_json     jsonb,
  explain_json     jsonb,
  artifact_uris    jsonb,
  created_at       timestamptz not null default now(),
  unique (run_id, event_id, method)
);

create index if not exists ensemble_forecasts_event_idx on public.ensemble_forecasts(event_id);
create index if not exists ensemble_forecasts_method_time_idx on public.ensemble_forecasts(method, created_at desc);
create index if not exists ensemble_forecasts_run_idx on public.ensemble_forecasts(run_id);

-- event_outcomes
create table if not exists public.event_outcomes (
  event_id           text primary key references public.events(event_id) on delete cascade,
  realized_ts        timestamptz not null,
  realized_label     text not null,
  realized_meta_json jsonb,
  created_at         timestamptz not null default now()
);

create index if not exists event_outcomes_realized_ts_idx on public.event_outcomes(realized_ts desc);
create index if not exists event_outcomes_label_idx on public.event_outcomes(realized_label);

-- weight_updates
create table if not exists public.weight_updates (
  id                  uuid primary key default gen_random_uuid(),
  asof_ts             timestamptz not null,
  context_key         text not null,
  method              text not null check (method in ('hedge','bayes','vw','stacked','blended')),
  weights_before_json jsonb not null,
  weights_after_json  jsonb not null,
  losses_json         jsonb,
  event_id            text references public.events(event_id) on delete set null,
  run_id              uuid references public.runs(run_id) on delete set null,
  created_at          timestamptz not null default now()
);

create index if not exists weight_updates_time_idx on public.weight_updates(asof_ts desc);
create index if not exists weight_updates_context_idx on public.weight_updates(context_key);
create index if not exists weight_updates_method_time_idx on public.weight_updates(method, asof_ts desc);

-- latest ensemble per event/method
create or replace view public.v_latest_ensemble_forecasts as
select distinct on (event_id, method)
  event_id, method, run_id, probs_json, confidence, uncertainty_json, weights_json, explain_json, created_at
from public.ensemble_forecasts
order by event_id, method, created_at desc;
