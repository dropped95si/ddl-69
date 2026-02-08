-- ddl-69: Ledger V2 patch (denormalized columns, outcome validation, updated_at triggers, RLS)

create or replace function public.tg_set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists runs_set_updated_at on public.runs;
create trigger runs_set_updated_at
before update on public.runs
for each row execute function public.tg_set_updated_at();

-- Denormalized columns for faster queries
alter table public.expert_forecasts
  add column if not exists subject_type text,
  add column if not exists subject_id text,
  add column if not exists event_type text,
  add column if not exists asof_ts timestamptz;

alter table public.ensemble_forecasts
  add column if not exists subject_type text,
  add column if not exists subject_id text,
  add column if not exists event_type text,
  add column if not exists asof_ts timestamptz;

update public.expert_forecasts ef
set subject_type = e.subject_type,
    subject_id   = e.subject_id,
    event_type   = e.event_type,
    asof_ts      = e.asof_ts
from public.events e
where ef.event_id = e.event_id
  and (ef.subject_id is null or ef.event_type is null or ef.asof_ts is null);

update public.ensemble_forecasts enf
set subject_type = e.subject_type,
    subject_id   = e.subject_id,
    event_type   = e.event_type,
    asof_ts      = e.asof_ts
from public.events e
where enf.event_id = e.event_id
  and (enf.subject_id is null or enf.event_type is null or enf.asof_ts is null);

create or replace function public.tg_sync_forecast_denorm_cols()
returns trigger
language plpgsql
as $$
declare
  e record;
begin
  select subject_type, subject_id, event_type, asof_ts
    into e
  from public.events
  where event_id = new.event_id;

  if not found then
    raise exception 'event_id % not found in events', new.event_id;
  end if;

  new.subject_type := e.subject_type;
  new.subject_id   := e.subject_id;
  new.event_type   := e.event_type;
  new.asof_ts      := e.asof_ts;

  return new;
end;
$$;

drop trigger if exists expert_forecasts_sync_denorm on public.expert_forecasts;
create trigger expert_forecasts_sync_denorm
before insert or update of event_id
on public.expert_forecasts
for each row execute function public.tg_sync_forecast_denorm_cols();

drop trigger if exists ensemble_forecasts_sync_denorm on public.ensemble_forecasts;
create trigger ensemble_forecasts_sync_denorm
before insert or update of event_id
on public.ensemble_forecasts
for each row execute function public.tg_sync_forecast_denorm_cols();

create index if not exists expert_forecasts_subject_time_idx
  on public.expert_forecasts (subject_id, asof_ts desc);
create index if not exists ensemble_forecasts_subject_time_idx
  on public.ensemble_forecasts (subject_id, asof_ts desc);
create index if not exists ensemble_forecasts_event_type_time_idx
  on public.ensemble_forecasts (event_type, asof_ts desc);

-- Outcome label validation
create or replace function public.tg_validate_event_outcome_label()
returns trigger
language plpgsql
as $$
declare
  et text;
begin
  select event_type into et
  from public.events
  where event_id = new.event_id;

  if not found then
    raise exception 'event_id % not found in events', new.event_id;
  end if;

  if et = 'state_event' then
    if new.realized_label not in ('REJECT','BREAK_FAIL','ACCEPT_CONTINUE') then
      raise exception 'Invalid realized_label % for state_event', new.realized_label;
    end if;
  elsif et = 'barrier_event' then
    if new.realized_label not in ('UPPER','LOWER','NONE') then
      raise exception 'Invalid realized_label % for barrier_event', new.realized_label;
    end if;
  else
    raise exception 'Unknown event_type % for event_id %', et, new.event_id;
  end if;

  return new;
end;
$$;

drop trigger if exists event_outcomes_validate_label on public.event_outcomes;
create trigger event_outcomes_validate_label
before insert or update of realized_label
on public.event_outcomes
for each row execute function public.tg_validate_event_outcome_label();

-- updated_at for other tables
alter table public.events add column if not exists updated_at timestamptz not null default now();
drop trigger if exists events_set_updated_at on public.events;
create trigger events_set_updated_at
before update on public.events
for each row execute function public.tg_set_updated_at();

alter table public.artifacts add column if not exists updated_at timestamptz not null default now();
drop trigger if exists artifacts_set_updated_at on public.artifacts;
create trigger artifacts_set_updated_at
before update on public.artifacts
for each row execute function public.tg_set_updated_at();

-- RLS policies
alter table public.runs enable row level security;
alter table public.events enable row level security;
alter table public.artifacts enable row level security;
alter table public.expert_forecasts enable row level security;
alter table public.ensemble_forecasts enable row level security;
alter table public.event_outcomes enable row level security;
alter table public.weight_updates enable row level security;

create or replace function public.jwt_role()
returns text
language sql
stable
as $$
  select coalesce(current_setting('request.jwt.claim.role', true), '');
$$;

-- Drop policies if re-run
DO $$
BEGIN
  EXECUTE 'drop policy if exists runs_read on public.runs';
  EXECUTE 'drop policy if exists runs_write_service on public.runs';
  EXECUTE 'drop policy if exists events_read on public.events';
  EXECUTE 'drop policy if exists events_write_service on public.events';
  EXECUTE 'drop policy if exists artifacts_read on public.artifacts';
  EXECUTE 'drop policy if exists artifacts_write_service on public.artifacts';
  EXECUTE 'drop policy if exists expert_forecasts_read on public.expert_forecasts';
  EXECUTE 'drop policy if exists expert_forecasts_write_service on public.expert_forecasts';
  EXECUTE 'drop policy if exists ensemble_forecasts_read on public.ensemble_forecasts';
  EXECUTE 'drop policy if exists ensemble_forecasts_write_service on public.ensemble_forecasts';
  EXECUTE 'drop policy if exists event_outcomes_read on public.event_outcomes';
  EXECUTE 'drop policy if exists event_outcomes_write_service on public.event_outcomes';
  EXECUTE 'drop policy if exists weight_updates_read on public.weight_updates';
  EXECUTE 'drop policy if exists weight_updates_write_service on public.weight_updates';
EXCEPTION WHEN OTHERS THEN
  NULL;
END $$;

create policy runs_read on public.runs
for select using (public.jwt_role() in ('authenticated','service_role'));
create policy runs_write_service on public.runs
for all using (public.jwt_role() = 'service_role')
with check (public.jwt_role() = 'service_role');

create policy events_read on public.events
for select using (public.jwt_role() in ('authenticated','service_role'));
create policy events_write_service on public.events
for all using (public.jwt_role() = 'service_role')
with check (public.jwt_role() = 'service_role');

create policy artifacts_read on public.artifacts
for select using (public.jwt_role() in ('authenticated','service_role'));
create policy artifacts_write_service on public.artifacts
for all using (public.jwt_role() = 'service_role')
with check (public.jwt_role() = 'service_role');

create policy expert_forecasts_read on public.expert_forecasts
for select using (public.jwt_role() in ('authenticated','service_role'));
create policy expert_forecasts_write_service on public.expert_forecasts
for all using (public.jwt_role() = 'service_role')
with check (public.jwt_role() = 'service_role');

create policy ensemble_forecasts_read on public.ensemble_forecasts
for select using (public.jwt_role() in ('authenticated','service_role'));
create policy ensemble_forecasts_write_service on public.ensemble_forecasts
for all using (public.jwt_role() = 'service_role')
with check (public.jwt_role() = 'service_role');

create policy event_outcomes_read on public.event_outcomes
for select using (public.jwt_role() in ('authenticated','service_role'));
create policy event_outcomes_write_service on public.event_outcomes
for all using (public.jwt_role() = 'service_role')
with check (public.jwt_role() = 'service_role');

create policy weight_updates_read on public.weight_updates
for select using (public.jwt_role() in ('authenticated','service_role'));
create policy weight_updates_write_service on public.weight_updates
for all using (public.jwt_role() = 'service_role')
with check (public.jwt_role() = 'service_role');
