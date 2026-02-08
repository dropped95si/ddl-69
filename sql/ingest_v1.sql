-- ddl-69: Normalized ingest schema (canonical market + text data)
-- Heavy time-series can be stored in Parquet; these tables are for
-- normalized, queryable slices, QC, and incremental ingestion control.

create extension if not exists pgcrypto;

-- Provider registry
create table if not exists public.data_providers (
  provider_id text primary key,
  description text,
  created_at  timestamptz not null default now()
);

insert into public.data_providers(provider_id, description)
values
  ('polygon','Polygon.io'),
  ('alpaca','Alpaca'),
  ('yahoo','Yahoo Finance'),
  ('csv','Local CSV'),
  ('webull','Webull')
on conflict (provider_id) do nothing;

-- Instruments (minimal; extend as needed)
create table if not exists public.instruments (
  instrument_id text primary key,     -- e.g., 'AAPL'
  instrument_type text not null default 'equity',
  exchange text,
  currency text,
  meta_json jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Watermarks: track incremental ingests by provider + dataset
create table if not exists public.provider_watermarks (
  id uuid primary key default gen_random_uuid(),
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  dataset text not null,              -- 'bars_1m','bars_1d','quotes','trades','lob'
  instrument_id text,                 -- nullable for global datasets
  last_ts timestamptz,
  last_cursor text,
  meta_json jsonb,
  updated_at timestamptz not null default now(),
  unique (provider_id, dataset, instrument_id)
);
create index if not exists provider_watermarks_lookup_idx
  on public.provider_watermarks (provider_id, dataset, instrument_id);

-- Data quality events (captures gaps, stale, schema anomalies)
create table if not exists public.data_quality_events (
  id uuid primary key default gen_random_uuid(),
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  dataset text not null,
  instrument_id text,
  asof_ts timestamptz not null,
  severity text not null check (severity in ('info','warn','error')),
  message text not null,
  meta_json jsonb,
  created_at timestamptz not null default now()
);
create index if not exists data_quality_time_idx on public.data_quality_events (asof_ts desc);
create index if not exists data_quality_instrument_idx on public.data_quality_events (instrument_id, asof_ts desc);

-- Canonical OHLCV bars (use for slices; store bulk in Parquet)
create table if not exists public.bars (
  instrument_id text not null references public.instruments(instrument_id) on delete cascade,
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  timeframe text not null,            -- '1m','5m','1h','1d'
  ts timestamptz not null,
  open double precision,
  high double precision,
  low double precision,
  close double precision,
  volume double precision,
  vwap double precision,
  trades_count integer,
  meta_json jsonb,
  primary key (instrument_id, provider_id, timeframe, ts)
);
create index if not exists bars_time_idx on public.bars (ts desc);
create index if not exists bars_instrument_time_idx on public.bars (instrument_id, timeframe, ts desc);

-- Trades (optional; keep only if you ingest granular data)
create table if not exists public.trades (
  instrument_id text not null references public.instruments(instrument_id) on delete cascade,
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  ts timestamptz not null,
  price double precision not null,
  size double precision,
  side text check (side in ('buy','sell','unknown')),
  exchange text,
  conditions text[],
  meta_json jsonb,
  primary key (instrument_id, provider_id, ts, price)
);
create index if not exists trades_instrument_time_idx on public.trades (instrument_id, ts desc);

-- Quotes (NBBO-ish)
create table if not exists public.quotes (
  instrument_id text not null references public.instruments(instrument_id) on delete cascade,
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  ts timestamptz not null,
  bid double precision,
  ask double precision,
  bid_size double precision,
  ask_size double precision,
  meta_json jsonb,
  primary key (instrument_id, provider_id, ts)
);
create index if not exists quotes_instrument_time_idx on public.quotes (instrument_id, ts desc);

-- L2/L3 Order book snapshots (optional)
create table if not exists public.lob_snapshots (
  instrument_id text not null references public.instruments(instrument_id) on delete cascade,
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  ts timestamptz not null,
  depth integer not null default 10,
  bids jsonb not null,  -- [{"px":...,"sz":...}, ...]
  asks jsonb not null,
  meta_json jsonb,
  primary key (instrument_id, provider_id, ts)
);
create index if not exists lob_snapshots_instrument_time_idx on public.lob_snapshots (instrument_id, ts desc);

-- Corporate actions (optional; used for normalization/adjustments)
create table if not exists public.corporate_actions (
  id uuid primary key default gen_random_uuid(),
  instrument_id text not null references public.instruments(instrument_id) on delete cascade,
  action_type text not null check (action_type in ('split','dividend','symbol_change','earnings')),
  effective_ts timestamptz not null,
  value_json jsonb,
  source text,
  created_at timestamptz not null default now()
);
create index if not exists corporate_actions_instrument_time_idx on public.corporate_actions (instrument_id, effective_ts desc);

-- News items (optional)
create table if not exists public.news_items (
  id uuid primary key default gen_random_uuid(),
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  ts timestamptz not null,
  title text,
  body text,
  url text,
  tickers text[],
  sentiment double precision,
  embedding vector(1536), -- requires pgvector; remove if not using
  meta_json jsonb,
  created_at timestamptz not null default now()
);
-- If you don't use pgvector, comment out the embedding line above.
create index if not exists news_time_idx on public.news_items (ts desc);

-- Social items (optional)
create table if not exists public.social_items (
  id uuid primary key default gen_random_uuid(),
  provider_id text not null references public.data_providers(provider_id) on delete cascade,
  ts timestamptz not null,
  source text, -- reddit/x/stocktwits/etc
  author text,
  text_content text,
  tickers text[],
  sentiment double precision,
  meta_json jsonb,
  created_at timestamptz not null default now()
);
create index if not exists social_time_idx on public.social_items (ts desc);

-- updated_at triggers
create or replace function public.tg_set_updated_at_ingest()
returns trigger language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end; $$;

drop trigger if exists instruments_set_updated_at on public.instruments;
create trigger instruments_set_updated_at
before update on public.instruments
for each row execute function public.tg_set_updated_at_ingest();

drop trigger if exists provider_watermarks_set_updated_at on public.provider_watermarks;
create trigger provider_watermarks_set_updated_at
before update on public.provider_watermarks
for each row execute function public.tg_set_updated_at_ingest();

