-- ============================================================
-- Normalized Ingestion Schema (multi-provider)
-- Canonical market + alt-data tables.
-- Heavy payloads -> Parquet, pointers in artifacts (ledger).
-- ============================================================

create extension if not exists pgcrypto;

-- Providers
create table if not exists public.providers (
  provider_id uuid primary key default gen_random_uuid(),
  name text not null unique,
  kind text not null check (kind in ('market','news','social','fundamentals','lob','custom')),
  priority int not null default 100,
  meta jsonb,
  created_at timestamptz not null default now()
);

-- Instruments
create table if not exists public.instruments (
  instrument_id uuid primary key default gen_random_uuid(),
  symbol text not null,
  venue text,
  asset_class text not null default 'equity',
  currency text default 'USD',
  meta jsonb,
  created_at timestamptz not null default now(),
  unique(symbol, coalesce(venue,''), asset_class)
);

-- Provider watermarks (last ingested point)
create table if not exists public.provider_watermarks (
  id uuid primary key default gen_random_uuid(),
  provider_id uuid not null references public.providers(provider_id) on delete cascade,
  instrument_id uuid not null references public.instruments(instrument_id) on delete cascade,
  feed text not null, -- bars|trades|quotes|lob|news|social
  granularity text,   -- 1m|5m|1d|tick|L2 etc
  last_ts timestamptz,
  last_seq bigint,
  status text not null default 'ok' check (status in ('ok','stale','error')),
  meta jsonb,
  updated_at timestamptz not null default now(),
  unique(provider_id, instrument_id, feed, coalesce(granularity,''))
);

create index if not exists provider_watermarks_lookup_idx
  on public.provider_watermarks (provider_id, instrument_id, feed, granularity);

-- Data quality per batch/run/provider (run_id references ledger if present)
create table if not exists public.data_quality (
  id uuid primary key default gen_random_uuid(),
  run_id uuid,
  provider_id uuid references public.providers(provider_id) on delete set null,
  instrument_id uuid references public.instruments(instrument_id) on delete set null,
  feed text not null,
  asof_ts timestamptz not null default now(),
  missingness double precision,
  stale_sec int,
  outlier_rate double precision,
  notes text,
  meta jsonb
);

create index if not exists data_quality_asof_idx on public.data_quality (asof_ts desc);

-- Canonical OHLCV bars
create table if not exists public.bars (
  instrument_id uuid not null references public.instruments(instrument_id) on delete cascade,
  provider_id uuid references public.providers(provider_id) on delete set null,
  ts timestamptz not null,
  timeframe text not null, -- 1m|5m|1h|1d
  open double precision,
  high double precision,
  low double precision,
  close double precision,
  volume double precision,
  vwap double precision,
  trade_count int,
  meta jsonb,
  primary key (instrument_id, timeframe, ts)
);

create index if not exists bars_time_idx on public.bars (ts desc);
create index if not exists bars_provider_idx on public.bars (provider_id, timeframe, ts desc);

-- Trades (optional)
create table if not exists public.trades (
  instrument_id uuid not null references public.instruments(instrument_id) on delete cascade,
  provider_id uuid references public.providers(provider_id) on delete set null,
  ts timestamptz not null,
  price double precision not null,
  size double precision,
  side text check (side in ('buy','sell','unknown')),
  conditions text[],
  trade_id text,
  meta jsonb
);

create index if not exists trades_inst_ts_idx on public.trades (instrument_id, ts desc);

-- Quotes/NBBO (optional)
create table if not exists public.quotes (
  instrument_id uuid not null references public.instruments(instrument_id) on delete cascade,
  provider_id uuid references public.providers(provider_id) on delete set null,
  ts timestamptz not null,
  bid_price double precision,
  bid_size double precision,
  ask_price double precision,
  ask_size double precision,
  quote_id text,
  meta jsonb
);

create index if not exists quotes_inst_ts_idx on public.quotes (instrument_id, ts desc);

-- L2 order book snapshots (optional; raw feed should go Parquet)
create table if not exists public.lob_snapshots (
  instrument_id uuid not null references public.instruments(instrument_id) on delete cascade,
  provider_id uuid references public.providers(provider_id) on delete set null,
  ts timestamptz not null,
  depth int not null default 10,
  bids jsonb not null,
  asks jsonb not null,
  meta jsonb,
  primary key (instrument_id, ts)
);

create index if not exists lob_snapshots_ts_idx on public.lob_snapshots (ts desc);

-- News items
create table if not exists public.news_items (
  news_id uuid primary key default gen_random_uuid(),
  provider_id uuid references public.providers(provider_id) on delete set null,
  published_ts timestamptz not null,
  title text,
  summary text,
  url text,
  tickers text[],
  sentiment double precision,
  relevance double precision,
  content_uri text,
  meta jsonb
);

create index if not exists news_published_idx on public.news_items (published_ts desc);
create index if not exists news_tickers_gin on public.news_items using gin (tickers);

-- Social items (Stocktwits/Reddit/X etc.)
create table if not exists public.social_items (
  post_id uuid primary key default gen_random_uuid(),
  provider_id uuid references public.providers(provider_id) on delete set null,
  published_ts timestamptz not null,
  source text,
  author text,
  tickers text[],
  text_uri text,
  sentiment double precision,
  intensity double precision,
  meta jsonb
);

create index if not exists social_published_idx on public.social_items (published_ts desc);
create index if not exists social_tickers_gin on public.social_items using gin (tickers);
