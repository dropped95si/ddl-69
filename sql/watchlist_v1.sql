-- Watchlist rankings for UI
create extension if not exists pgcrypto;

create table if not exists public.watchlist_rankings (
  id uuid primary key default gen_random_uuid(),
  asof_ts timestamptz not null,
  ticker text not null,
  score double precision not null,
  p_accept double precision not null,
  label text,
  plan_type text,
  weights_json jsonb,
  meta_json jsonb,
  created_at timestamptz not null default now(),
  unique (asof_ts, ticker)
);

create index if not exists watchlist_rankings_asof_idx on public.watchlist_rankings (asof_ts desc);
create index if not exists watchlist_rankings_ticker_idx on public.watchlist_rankings (ticker);
