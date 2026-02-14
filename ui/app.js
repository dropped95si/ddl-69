// PRIMARY SOURCES - Real trading system (Supabase + ML Pipeline)
const DEFAULT_WATCHLIST = "/api/live";           // Supabase ensemble predictions (strict mode)
const DEFAULT_FORECASTS = "/api/forecasts";      // Ensemble forecast time series
const DEFAULT_FINVIZ = "/api/finviz?mode=swing&count=100";  // TP/SL heuristics
const DEFAULT_OVERLAY = "/api/overlays";          // Technical overlay charts
const DEFAULT_WALKFORWARD = "/api/walkforward";  // Walk-forward backtesting
const DEFAULT_NEWS = "/api/news";                // News + sentiment from Supabase/Polygon
const DEFAULT_PROJECTION = "/api/projection";

const watchlistInput = document.getElementById("watchlistUrl");
const newsInput = document.getElementById("newsUrl");
const overlayInput = document.getElementById("overlayUrl");
const walkforwardInput = document.getElementById("walkforwardUrl");
const autoRefreshInput = document.getElementById("autoRefreshSec");
const refreshBtn = document.getElementById("refreshBtn");
const saveBtn = document.getElementById("saveBtn");
const topNInput = document.getElementById("topN");

const asofValue = document.getElementById("asofValue");
const sourceValue = document.getElementById("sourceValue");
const countValue = document.getElementById("countValue");
const watchlistMeta = document.getElementById("watchlistMeta");
const watchlistGrid = document.getElementById("watchlistGrid");
const watchlistTable = document.getElementById("watchlistTable");
const watchlistFilter = document.getElementById("watchlistFilter");
const watchlistSort = document.getElementById("watchlistSort");
const marketCapFilter = document.getElementById("marketCapFilter");
const assetTypeFilter = document.getElementById("assetTypeFilter");
const denseCardsToggle = document.getElementById("denseCards");
const compactWeightsToggle = document.getElementById("compactWeights");
const clearFilterBtn = document.getElementById("clearFilter");
const gridViewBtn = document.getElementById("gridViewBtn");
const tableViewBtn = document.getElementById("tableViewBtn");
const scoreBars = document.getElementById("scoreBars");
const newsMeta = document.getElementById("newsMeta");
const newsGrid = document.getElementById("newsGrid");
const dataStatus = document.getElementById("dataStatus");
const lastRefresh = document.getElementById("lastRefresh");
const walkforwardMeta = document.getElementById("walkforwardMeta");
const walkforwardGrid = document.getElementById("walkforwardGrid");
const timeframeSel = document.getElementById("timeframeSel");
const runSel = document.getElementById("runSel");
const runMeta = document.getElementById("runMeta");

const modal = document.getElementById("symbolModal");
const modalClose = document.getElementById("modalClose");
const modalSymbol = document.getElementById("modalSymbol");
const modalScore = document.getElementById("modalScore");
const modalProb = document.getElementById("modalProb");
const modalLabel = document.getElementById("modalLabel");
const modalWeights = document.getElementById("modalWeights");
const tvChart = document.getElementById("tvChart");
const detailSymbol = document.getElementById("detailSymbol");
const detailScore = document.getElementById("detailScore");
const detailProb = document.getElementById("detailProb");
const detailLabel = document.getElementById("detailLabel");
const detailWeights = document.getElementById("detailWeights");
const weightsFilter = document.getElementById("weightsFilter");
const detailWeightStats = document.getElementById("detailWeightStats");
const detailChart = document.getElementById("detailChart");
const engineGrid = document.getElementById("engineGrid");
const scenarioCard = document.getElementById("scenarioCard");
const detailProjectionChart = document.getElementById("detailProjectionChart");
const detailProjectionMeta = document.getElementById("detailProjectionMeta");
const detailProjectionTargets = document.getElementById("detailProjectionTargets");
const detailOverlayChart = document.getElementById("detailOverlayChart");
const detailOverlayMeta = document.getElementById("detailOverlayMeta");
const detailOverlaySummary = document.getElementById("detailOverlaySummary");
const detailCopyBtn = document.getElementById("detailCopyBtn");
const detailTvBtn = document.getElementById("detailTvBtn");

function normalizeStoredUrl(value, fallback, bannedSubstrings = []) {
  const txt = String(value || "").trim();
  if (!txt) return fallback;
  for (const banned of bannedSubstrings) {
    if (txt.includes(banned)) return fallback;
  }
  return txt;
}

const rawStoredWatchlist = localStorage.getItem("ddl69_watchlist_url");
const storedWatchlist = normalizeStoredUrl(rawStoredWatchlist, DEFAULT_WATCHLIST, [
  "/api/finviz",
  "/api/demo",
  "watchlist.json",
]);
if (rawStoredWatchlist && storedWatchlist !== rawStoredWatchlist) {
  localStorage.setItem("ddl69_watchlist_url", storedWatchlist);
}

const rawStoredNews = localStorage.getItem("ddl69_news_url");
const storedNews = normalizeStoredUrl(rawStoredNews, DEFAULT_NEWS, [
  "/storage/v1/object/public/artifacts/news/polygon_news_",
  "news.json",
]);
if (rawStoredNews && storedNews !== rawStoredNews) {
  localStorage.setItem("ddl69_news_url", storedNews);
}

const rawStoredOverlay = localStorage.getItem("ddl69_overlay_url");
const storedOverlay = normalizeStoredUrl(rawStoredOverlay, DEFAULT_OVERLAY, [
  "overlay.json",
  "/storage/v1/object/public/artifacts/overlays/",
]);
if (rawStoredOverlay && storedOverlay !== rawStoredOverlay) {
  localStorage.setItem("ddl69_overlay_url", storedOverlay);
}

const storedSort = normalizeWatchlistSortMode(localStorage.getItem("ddl69_watchlist_sort") || "score_desc");
const storedMarketCapFilter = localStorage.getItem("ddl69_market_cap_filter") || "all";
const storedAssetTypeFilter = localStorage.getItem("ddl69_asset_type_filter") || "all";
const storedAutoRefresh = localStorage.getItem("ddl69_autorefresh_sec") || "300";
const storedDense = localStorage.getItem("ddl69_dense_cards") || "0";
const storedView = localStorage.getItem("ddl69_watchlist_view") || "grid";
const storedCompactWeights = localStorage.getItem("ddl69_compact_weights") || "0";
const storedWeightsFilter = localStorage.getItem("ddl69_weights_filter") || "top";
const storedRunId = localStorage.getItem("ddl69_run_id") || "";
const rawStoredWalkforward = localStorage.getItem("ddl69_walkforward_url");
const storedWalkforward = normalizeStoredUrl(rawStoredWalkforward, DEFAULT_WALKFORWARD, [
  "wfo.json",
  "walkforward.json",
  "/storage/v1/object/public/artifacts/walkforward/",
]);
if (rawStoredWalkforward && storedWalkforward !== rawStoredWalkforward) {
  localStorage.setItem("ddl69_walkforward_url", storedWalkforward);
}
if (watchlistInput) watchlistInput.value = storedWatchlist;
if (newsInput) newsInput.value = storedNews;
if (overlayInput) overlayInput.value = storedOverlay;
if (walkforwardInput) walkforwardInput.value = storedWalkforward;
if (watchlistSort) {
  watchlistSort.value = storedSort;
  if (!watchlistSort.value) watchlistSort.value = "score_desc";
  localStorage.setItem("ddl69_watchlist_sort", watchlistSort.value);
}
if (marketCapFilter) marketCapFilter.value = storedMarketCapFilter;
if (assetTypeFilter) assetTypeFilter.value = storedAssetTypeFilter;
if (autoRefreshInput) autoRefreshInput.value = storedAutoRefresh;
if (denseCardsToggle) denseCardsToggle.checked = storedDense === "1";
if (compactWeightsToggle) compactWeightsToggle.checked = storedCompactWeights === "1";
if (weightsFilter) weightsFilter.value = storedWeightsFilter;
if (runSel) runSel.value = storedRunId;
setWatchlistView(storedView);

let overlayData = null;
let lastWatchlistData = null;
let lastNewsData = null;
let walkforwardData = null;
let runCatalog = [];
let currentRunId = String(storedRunId || "").trim();
let currentDetailRow = null;
let autoRefreshTimer = null;
let autoRefreshHeartbeat = null;
let refreshInFlight = false;
let pendingRefresh = false;
let lastSuccessfulRefreshMs = 0;
let projectionChart = null;
let projectionResizeObserver = null;
let projectionReqToken = 0;
let overlayReqToken = 0;

function debounce(fn, delay = 350) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), delay);
  };
}

function requestRefresh() {
  if (refreshInFlight) {
    pendingRefresh = true;
    return;
  }
  refreshAll();
}

const refreshSoon = debounce(() => {
  requestRefresh();
}, 250);

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeUrl(value) {
  if (!value) return "#";
  try {
    const url = new URL(value);
    if (url.protocol === "http:" || url.protocol === "https:") return url.href;
  } catch (err) {
    // fall through
  }
  return "#";
}

function withQueryParam(rawUrl, key, value) {
  const input = String(rawUrl || "").trim();
  if (!input) return "";
  try {
    const parsed = new URL(input, window.location.origin);
    parsed.searchParams.set(key, value);
    if (/^https?:\/\//i.test(input)) return parsed.toString();
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch (err) {
    const hasQuery = input.includes("?");
    const sep = hasQuery ? "&" : "?";
    return `${input}${sep}${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
  }
}

function formatDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toISOString().replace("T", " ").replace("Z", " UTC");
}

function formatDateShort(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toISOString().slice(0, 10);
}

function parseHorizonDaysValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const txt = String(value || "").trim().toLowerCase();
  if (!txt) return null;
  const m = txt.match(/^([0-9]+(?:\.[0-9]+)?)\s*(d|day|days|w|wk|week|weeks|mo|mon|month|months|m|y|yr|year|years)?$/);
  if (!m) return null;
  const v = Number(m[1]);
  if (!Number.isFinite(v)) return null;
  const unit = m[2] || "d";
  if (unit === "d" || unit === "day" || unit === "days") return v;
  if (unit === "w" || unit === "wk" || unit === "week" || unit === "weeks") return v * 7;
  if (unit === "mo" || unit === "mon" || unit === "month" || unit === "months" || unit === "m") return v * 30;
  if (unit === "y" || unit === "yr" || unit === "year" || unit === "years") return v * 365;
  return null;
}

function getProjectedExitDate(row) {
  if (!row || typeof row !== "object") return "—";
  const meta = getMeta(row);
  const horizon =
    parseHorizonDaysValue(row.horizon_days) ??
    parseHorizonDaysValue(meta.horizon_days) ??
    parseHorizonDaysValue(meta.horizon);
  if (!Number.isFinite(horizon) || horizon <= 0) return "—";

  const asofRaw = row.created_at || row.asof || meta.asof || null;
  const base = asofRaw ? new Date(asofRaw) : new Date();
  if (Number.isNaN(base.getTime())) return "—";

  const projected = new Date(base.getTime() + Math.round(horizon) * 24 * 60 * 60 * 1000);
  if (Number.isNaN(projected.getTime())) return "—";
  return projected.toISOString().slice(0, 10);
}

function formatPct(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

function parseMarketCap(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  if (typeof raw === "number" && Number.isFinite(raw) && raw > 0) return raw;
  const txt = String(raw).trim().toUpperCase().replace(/[$,\s]/g, "");
  if (!txt) return null;
  const m = txt.match(/^([0-9]+(?:\.[0-9]+)?)([KMBT])?$/);
  if (!m) return null;
  const value = Number(m[1]);
  if (!Number.isFinite(value) || value <= 0) return null;
  const unit = m[2] || "";
  if (unit === "T") return value * 1_000_000_000_000;
  if (unit === "B") return value * 1_000_000_000;
  if (unit === "M") return value * 1_000_000;
  if (unit === "K") return value * 1_000;
  return value;
}

function classifyCapBucketFromValue(capValue) {
  if (!Number.isFinite(capValue) || capValue <= 0) return "unknown";
  if (capValue >= 200_000_000_000) return "mega";
  if (capValue >= 10_000_000_000) return "large";
  if (capValue >= 2_000_000_000) return "mid";
  if (capValue >= 300_000_000) return "small";
  return "micro";
}

function normalizeCapBucket(rawBucket) {
  const txt = String(rawBucket || "").trim().toLowerCase().replace(/[\s_-]+/g, "");
  if (!txt) return "";
  if (txt.startsWith("mega")) return "mega";
  if (txt.startsWith("large") || txt === "big") return "large";
  if (txt.startsWith("mid")) return "mid";
  if (txt.startsWith("small")) return "small";
  if (txt.startsWith("micro") || txt.startsWith("nano")) return "micro";
  if (txt === "unknown" || txt === "na" || txt === "n/a") return "unknown";
  return "";
}

function normalizeAssetType(rawType) {
  const txt = String(rawType || "").trim().toLowerCase();
  if (!txt) return "unknown";
  if (txt.includes("etf") || txt.includes("etn") || txt.includes("fund")) return "etf";
  if (txt.includes("equity") || txt.includes("stock") || txt.includes("common")) return "equity";
  if (txt === "unknown" || txt === "na" || txt === "n/a") return "unknown";
  return txt;
}

function getRowMarketCap(row) {
  const meta = getMeta(row);
  const raw =
    row?.market_cap ??
    row?.marketCap ??
    meta.market_cap ??
    meta.marketCap ??
    meta.market_capitalization ??
    meta.market_cap_usd ??
    null;
  return parseMarketCap(raw);
}

function getRowCapBucket(row) {
  const meta = getMeta(row);
  const direct =
    normalizeCapBucket(row?.cap_bucket) ||
    normalizeCapBucket(row?.capBucket) ||
    normalizeCapBucket(meta.cap_bucket) ||
    normalizeCapBucket(meta.capBucket) ||
    normalizeCapBucket(meta.market_cap_bucket);
  if (direct) return direct;
  return classifyCapBucketFromValue(getRowMarketCap(row));
}

function getRowAssetType(row) {
  const meta = getMeta(row);
  return normalizeAssetType(
    row?.asset_type ??
    row?.assetType ??
    meta.asset_type ??
    meta.assetType ??
    meta.security_type ??
    meta.quote_type ??
    "unknown"
  );
}

function formatMarketCap(capValue) {
  if (!Number.isFinite(capValue) || capValue <= 0) return "—";
  if (capValue >= 1_000_000_000_000) return `$${(capValue / 1_000_000_000_000).toFixed(2)}T`;
  if (capValue >= 1_000_000_000) return `$${(capValue / 1_000_000_000).toFixed(1)}B`;
  if (capValue >= 1_000_000) return `$${(capValue / 1_000_000).toFixed(1)}M`;
  if (capValue >= 1_000) return `$${(capValue / 1_000).toFixed(1)}K`;
  return `$${capValue.toFixed(0)}`;
}

function formatCapBucket(bucket) {
  if (bucket === "mega") return "Mega";
  if (bucket === "large") return "Large";
  if (bucket === "mid") return "Mid";
  if (bucket === "small") return "Small";
  if (bucket === "micro") return "Micro";
  return "Unknown";
}

function capFilterMatch(row, capFilterMode) {
  const mode = String(capFilterMode || "all").toLowerCase();
  if (mode === "all") return true;
  const bucket = getRowCapBucket(row);
  if (mode === "unknown") return bucket === "unknown";
  return bucket === mode;
}

function assetFilterMatch(row, assetMode) {
  const mode = String(assetMode || "all").toLowerCase();
  if (mode === "all") return true;
  const assetType = getRowAssetType(row);
  return assetType === mode;
}

function normalizeWatchlistSortMode(rawMode) {
  const mode = String(rawMode || "").trim().toLowerCase();
  if (!mode) return "score_desc";
  if (mode === "score") return "score_desc";
  if (mode === "accept") return "accept_desc";
  if (mode === "ticker") return "ticker_asc";
  if (mode === "smallcap") return "market_cap_asc";
  if (mode === "market_cap") return "market_cap_desc";
  const allowed = new Set([
    "score_desc",
    "score_asc",
    "accept_desc",
    "accept_asc",
    "reject_desc",
    "reject_asc",
    "confidence_desc",
    "confidence_asc",
    "expected_return_desc",
    "expected_return_asc",
    "ticker_asc",
    "ticker_desc",
    "market_cap_desc",
    "market_cap_asc",
    "horizon_desc",
    "horizon_asc",
  ]);
  if (allowed.has(mode)) return mode;
  return "score_desc";
}

function getWatchlistSortSpec(rawMode) {
  const normalized = normalizeWatchlistSortMode(rawMode);
  let key = normalized;
  let dir = "desc";
  if (normalized.endsWith("_asc")) {
    key = normalized.slice(0, -4);
    dir = "asc";
  } else if (normalized.endsWith("_desc")) {
    key = normalized.slice(0, -5);
    dir = "desc";
  }
  return {
    mode: normalized,
    key,
    dir,
  };
}

function getSortValueForRow(row, key) {
  if (key === "ticker") {
    return String(row?.ticker || row?.symbol || "").toUpperCase();
  }
  if (key === "score") return Number(row?.score ?? row?.p_accept ?? 0);
  if (key === "accept") return Number(row?.p_accept ?? 0);
  if (key === "reject") return Number(row?.p_reject ?? 0);
  if (key === "confidence") return Number(row?.confidence ?? 0);
  if (key === "expected_return") {
    if (row?.expected_return_pct !== undefined && row?.expected_return_pct !== null) {
      return Number(row.expected_return_pct);
    }
    const base = Number(row?.expected_return ?? 0);
    return Number.isFinite(base) ? base * 100 : 0;
  }
  if (key === "market_cap") {
    const cap = getRowMarketCap(row);
    if (Number.isFinite(cap) && cap > 0) return cap;
    const bucket = getRowCapBucket(row);
    if (bucket === "mega") return 400_000_000_000;
    if (bucket === "large") return 20_000_000_000;
    if (bucket === "mid") return 5_000_000_000;
    if (bucket === "small") return 1_000_000_000;
    if (bucket === "micro") return 100_000_000;
    return null;
  }
  if (key === "horizon") return parseHorizonDaysValue(row?.horizon_days ?? getMeta(row)?.horizon ?? null);
  return Number(row?.score ?? row?.p_accept ?? 0);
}

function compareWatchlistRows(a, b, sortSpec) {
  const spec = sortSpec && typeof sortSpec === "object" ? sortSpec : getWatchlistSortSpec(sortSpec);
  const dirFactor = spec.dir === "asc" ? 1 : -1;
  if (spec.key === "ticker") {
    const at = String(getSortValueForRow(a, "ticker") || "");
    const bt = String(getSortValueForRow(b, "ticker") || "");
    const cmp = at.localeCompare(bt);
    if (cmp !== 0) return cmp * dirFactor;
  } else {
    const avRaw = getSortValueForRow(a, spec.key);
    const bvRaw = getSortValueForRow(b, spec.key);
    const av = Number(avRaw);
    const bv = Number(bvRaw);
    const aMissing = !Number.isFinite(av);
    const bMissing = !Number.isFinite(bv);
    if (aMissing && !bMissing) return 1;
    if (!aMissing && bMissing) return -1;
    if (!aMissing && !bMissing && av !== bv) {
      return (av > bv ? 1 : -1) * dirFactor;
    }
  }
  const scoreCmp = Number(b?.score ?? b?.p_accept ?? 0) - Number(a?.score ?? a?.p_accept ?? 0);
  if (scoreCmp !== 0) return scoreCmp;
  return String(a?.ticker || a?.symbol || "").localeCompare(String(b?.ticker || b?.symbol || ""));
}

function describeWatchlistSort(spec) {
  if (!spec || typeof spec !== "object") return "Score (desc)";
  const keyNames = {
    score: "Score",
    accept: "P(accept)",
    reject: "P(reject)",
    confidence: "Confidence",
    expected_return: "Expected Return",
    ticker: "Ticker",
    market_cap: "Market Cap",
    horizon: "Horizon",
  };
  const keyName = keyNames[spec.key] || spec.key;
  return `${keyName} (${spec.dir})`;
}

function setWatchlistSort(mode, options = {}) {
  const normalized = normalizeWatchlistSortMode(mode);
  if (watchlistSort) watchlistSort.value = normalized;
  if (options.persist !== false) {
    localStorage.setItem("ddl69_watchlist_sort", normalized);
  }
  if (options.rerender !== false && lastWatchlistData) {
    renderWatchlist(lastWatchlistData);
  }
}

function parseTickersList(value) {
  if (Array.isArray(value)) {
    return value
      .map((v) => String(v || "").trim().toUpperCase())
      .filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/[,\s]+/)
      .map((v) => v.trim().toUpperCase())
      .filter(Boolean);
  }
  return [];
}

function normalizeNewsItems(data) {
  const raw = data?.results || data?.data || data?.items || data?.news || data;
  if (!Array.isArray(raw)) return [];
  const seen = new Set();
  const items = [];
  raw.forEach((item) => {
    if (!item || typeof item !== "object") return;
    const title = String(item.title || item.headline || "").trim();
    const url = safeUrl(item.article_url || item.url || "");
    const key = `${title.toLowerCase()}|${url}`;
    if (seen.has(key)) return;
    seen.add(key);
    const tickers = parseTickersList(item.tickers || item.symbols || item.ticker || "");
    const published =
      item.published_utc ||
      item.published ||
      item.published_at ||
      item.timestamp ||
      item.time ||
      item.date ||
      null;
    const sentimentRaw = item.sentiment_score ?? item.sentiment ?? item.score;
    const sentimentNum = Number(sentimentRaw);
    items.push({
      title: title || "Untitled",
      url: url || "#",
      tickers,
      published,
      sentiment: Number.isFinite(sentimentNum) ? sentimentNum : null,
      source: item.source || item.publisher_name || item.publisher?.name || "",
    });
  });
  items.sort((a, b) => {
    const ta = new Date(a.published || 0).getTime();
    const tb = new Date(b.published || 0).getTime();
    return (Number.isFinite(tb) ? tb : 0) - (Number.isFinite(ta) ? ta : 0);
  });
  return items;
}
function getMeta(row) {
  return row.meta || row.meta_json || {};
}

function getTargetPrice(row) {
  const meta = getMeta(row);
  return (
    row.target_price ??
    meta.target_price ??
    meta.price_target ??
    meta.plan_target ??
    meta.next_level ??
    null
  );
}

function getHitProbability(row) {
  const meta = getMeta(row);
  const val =
    row.p_hit ??
    row.p_target ??
    meta.p_hit ??
    meta.p_target ??
    meta.p_price_hit ??
    row.p_accept ??
    0;
  return Number(val) || 0;
}

function normalizeLevel(val) {
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

function getScenarios(row) {
  const meta = getMeta(row);
  const upTps = [meta.tp1, meta.tp2, meta.tp3, meta.target_price, row.target_price]
    .map(normalizeLevel)
    .filter((n) => n !== null);
  const downTps = [meta.tp1_down, meta.tp2_down, meta.tp3_down].map(normalizeLevel).filter((n) => n !== null);
  const upSls = [meta.sl1, meta.sl2, meta.sl3].map(normalizeLevel).filter((n) => n !== null);
  const downSls = [meta.sl1_down, meta.sl2_down, meta.sl3_down].map(normalizeLevel).filter((n) => n !== null);

  const pUp = Number(meta.p_up ?? meta.p_bounce ?? meta.p_long ?? row.p_accept ?? 0);
  const pDown = Number(meta.p_down ?? meta.p_drop ?? 1 - (row.p_accept ?? 0));

  const horizon = meta.horizon || row.plan_type || "swing";

  return {
    up: {
      label: "Bounce / Up",
      prob: pUp,
      eta: meta.eta_up || meta.eta || meta.tp_eta || null,
      horizon,
      tps: upTps,
      sls: upSls,
    },
    down: {
      label: "Break / Down",
      prob: pDown,
      eta: meta.eta_down || meta.sl_eta || null,
      horizon,
      tps: downTps,
      sls: downSls,
    },
  };
}

function renderScenarioCard(row, scenarios = null) {
  const s = scenarios || getScenarios(row);
  const lastPrice = normalizeLevel(getMeta(row).last_price);
  const chip = (label, value, accent = "") =>
    `<div class="prob-chip ${accent}"><span>${label}</span><span>${value}</span></div>`;

  const block = (title, data, accent) => {
    const probTxt = formatPct(data.prob || 0, 1);
    const etaTxt = formatDateShort(data.eta);
    const horizon = data.horizon || "—";
    const tpList =
      data.tps.length === 0
        ? `<div class="small-note">No targets</div>`
        : data.tps
            .slice(0, 3)
            .map((v, i) => `<div class="level-pill tp">TP${i + 1}: $${Number(v).toFixed(2)}</div>`)
            .join("");
    const slList =
      data.sls.length === 0
        ? `<div class="small-note">No stops</div>`
        : data.sls
            .slice(0, 3)
            .map((v, i) => `<div class="level-pill sl">SL${i + 1}: $${Number(v).toFixed(2)}</div>`)
            .join("");
    return `
      <div class="scenario-block ${accent}">
        <div class="scenario-head">
          <div>
            <div class="mini-label">${escapeHtml(title)}</div>
            <div class="scenario-prob">${probTxt}</div>
          </div>
          <div class="scenario-meta">
            ${chip("ETA", etaTxt)}
            ${chip("Horizon", escapeHtml(horizon))}
          </div>
        </div>
        <div class="scenario-levels">${tpList}${slList}</div>
        <div class="scenario-foot">
          ${lastPrice ? `<span class="small-note">Ref price: $${lastPrice.toFixed(2)}</span>` : ""}
          <span class="small-note">Confidence bands shown on chart</span>
        </div>
      </div>
    `;
  };

  return `
    <div class="scenario-grid">
      ${block("Upside path", s.up, "accent-up")}
      ${block("Downside path", s.down, "accent-down")}
    </div>
  `;
}

function toUnixSeconds(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") {
    if (value > 1e12) return Math.floor(value / 1000);
    if (value > 1e10) return Math.floor(value / 1000);
    if (value > 1e9) return Math.floor(value);
    return Math.floor(value);
  }
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return null;
  return Math.floor(d.getTime() / 1000);
}

function normalizeOverlayData(raw) {
  if (!raw) return null;
  if (raw.symbols && typeof raw.symbols === "object") {
    const symbols = {};
    Object.entries(raw.symbols).forEach(([key, value]) => {
      symbols[String(key).toUpperCase()] = value;
    });
    return { ...raw, symbols };
  }
  if (raw.symbol) {
    return { asof: raw.asof, symbols: { [String(raw.symbol).toUpperCase()]: raw } };
  }
  if (Array.isArray(raw)) {
    const symbols = {};
    raw.forEach((entry) => {
      if (entry && entry.symbol) symbols[String(entry.symbol).toUpperCase()] = entry;
    });
    return { symbols };
  }
  return { symbols: {} };
}

function getOverlayForSymbol(symbol) {
  if (!overlayData || !symbol) return null;
  const key = String(symbol).toUpperCase();
  if (overlayData.symbols && overlayData.symbols[key]) return overlayData.symbols[key];
  if (overlayData.symbol && String(overlayData.symbol).toUpperCase() === key) return overlayData;
  if (overlayData.symbols && overlayData.symbols[symbol]) return overlayData.symbols[symbol];
  return null;
}

function mergeOverlayData(existing, incomingRaw) {
  const incoming = normalizeOverlayData(incomingRaw);
  if (!incoming) return existing;
  if (!existing || !existing.symbols) return incoming;
  return {
    ...existing,
    ...incoming,
    symbols: {
      ...(existing.symbols || {}),
      ...(incoming.symbols || {}),
    },
  };
}

async function loadOverlayForSymbol(symbol, row = null, scopeOverride = null) {
  if (!symbol || symbol === "—") return;
  if (getOverlayForSymbol(symbol)) {
    renderOverlayChart(symbol, row);
    return;
  }
  const overlayUrlRaw = (overlayInput ? overlayInput.value : DEFAULT_OVERLAY).trim();
  if (!overlayUrlRaw) {
    renderOverlayChart(symbol, row);
    return;
  }

  const token = ++overlayReqToken;
  const scope = normalizeScope(scopeOverride || resolveScopeForRow(row));
  const mode = scopeToMode(scope);
  let url = withQueryParam(overlayUrlRaw, "mode", mode);
  url = withQueryParam(url, "symbols", symbol);
  url = withQueryParam(url, "count", "1");

  if (detailOverlayMeta) {
    detailOverlayMeta.textContent = `Loading overlay for ${symbol} (${mode})...`;
  }
  try {
    const payload = await fetchJson(url);
    if (token !== overlayReqToken) return;
    overlayData = mergeOverlayData(overlayData, payload);
    renderOverlayChart(symbol, row);
  } catch (err) {
    if (token !== overlayReqToken) return;
    renderOverlayChart(symbol, row);
    if (detailOverlayMeta) {
      detailOverlayMeta.textContent = `Overlay error: ${err?.message || "Failed to load overlay"}`;
    }
  }
}

function resolveLineStyle(style) {
  const lw = window.LightweightCharts;
  if (!lw || !lw.LineStyle) return 0;
  const s = String(style || "solid").toLowerCase();
  if (s === "dotted") return lw.LineStyle.Dotted;
  if (s === "dashed") return lw.LineStyle.Dashed;
  return lw.LineStyle.Solid;
}

function normalizeProjectionSeries(rawSeries) {
  if (!Array.isArray(rawSeries)) return [];
  return rawSeries
    .map((pt) => {
      const time = toUnixSeconds(pt?.time ?? pt?.t ?? pt?.date ?? pt?.timestamp);
      const value = Number(pt?.value ?? pt?.close ?? pt?.price ?? pt?.v);
      if (!time || Number.isNaN(value)) return null;
      return { time, value };
    })
    .filter(Boolean);
}

function resetProjectionChart(invalidateRequest = false) {
  if (invalidateRequest) projectionReqToken += 1;
  if (projectionResizeObserver) {
    projectionResizeObserver.disconnect();
    projectionResizeObserver = null;
  }
  projectionChart = null;
  if (detailProjectionChart) detailProjectionChart.innerHTML = "";
}

function renderProjectionTargets(targets) {
  if (!detailProjectionTargets) return;
  if (!targets || typeof targets !== "object") {
    detailProjectionTargets.innerHTML = "";
    return;
  }

  const items = [
    ["TP1", targets.tp1, "tp"],
    ["TP2", targets.tp2, "tp"],
    ["TP3", targets.tp3, "tp"],
    ["SL1", targets.sl1, "sl"],
    ["SL2", targets.sl2, "sl"],
    ["SL3", targets.sl3, "sl"],
  ].filter(([, value]) => Number.isFinite(Number(value)));

  if (!items.length) {
    detailProjectionTargets.innerHTML = "";
    return;
  }

  detailProjectionTargets.innerHTML = items
    .map(
      ([label, value, cls]) =>
        `<div class="target-pill ${cls}"><span>${label}</span><span>$${Number(value).toFixed(2)}</span></div>`
    )
    .join("");
}

function renderProjectionState(message, targets = null) {
  if (detailProjectionMeta) detailProjectionMeta.textContent = message;
  renderProjectionTargets(targets);
}

function renderProjectionChart(data, row = null) {
  if (!detailProjectionChart) return;

  const lw = window.LightweightCharts;
  if (!lw) {
    resetProjectionChart();
    renderProjectionState("Projection chart library not loaded.");
    return;
  }

  const history = normalizeProjectionSeries(data?.history);
  const projection = data?.projection || {};
  const median = normalizeProjectionSeries(projection.median);
  const upper80 = normalizeProjectionSeries(projection.upper_80);
  const lower80 = normalizeProjectionSeries(projection.lower_80);
  const upper95 = normalizeProjectionSeries(projection.upper_95);
  const lower95 = normalizeProjectionSeries(projection.lower_95);
  const currentPrice = Number(data?.current_price);
  const targets = data?.targets || null;

  if (!history.length && !median.length) {
    resetProjectionChart();
    renderProjectionState("Projection data unavailable for this symbol.");
    return;
  }

  resetProjectionChart();
  projectionChart = lw.createChart(detailProjectionChart, {
    layout: {
      background: { type: "solid", color: "#070b15" },
      textColor: "#c5d0e6",
      fontFamily: "Space Grotesk, sans-serif",
    },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.05)" },
      horzLines: { color: "rgba(255,255,255,0.05)" },
    },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.1)" },
    timeScale: { borderColor: "rgba(255,255,255,0.1)" },
    height: detailProjectionChart.clientHeight || 300,
  });

  if (history.length) {
    const historySeries = projectionChart.addLineSeries({
      color: "#9fb2d3",
      lineWidth: 2,
    });
    historySeries.setData(history);
  }

  if (upper95.length) {
    const s = projectionChart.addLineSeries({
      color: "rgba(255, 143, 163, 0.55)",
      lineWidth: 1,
      lineStyle: lw.LineStyle.Dashed,
    });
    s.setData(upper95);
  }
  if (lower95.length) {
    const s = projectionChart.addLineSeries({
      color: "rgba(255, 143, 163, 0.55)",
      lineWidth: 1,
      lineStyle: lw.LineStyle.Dashed,
    });
    s.setData(lower95);
  }

  if (upper80.length) {
    const s = projectionChart.addLineSeries({
      color: "rgba(255, 212, 121, 0.72)",
      lineWidth: 1,
      lineStyle: lw.LineStyle.Dotted,
    });
    s.setData(upper80);
  }
  if (lower80.length) {
    const s = projectionChart.addLineSeries({
      color: "rgba(255, 212, 121, 0.72)",
      lineWidth: 1,
      lineStyle: lw.LineStyle.Dotted,
    });
    s.setData(lower80);
  }

  if (median.length) {
    const medianSeries = projectionChart.addLineSeries({
      color: "#12d6ff",
      lineWidth: 3,
    });
    medianSeries.setData(median);
  }

  const refSeries = projectionChart.addLineSeries({
    color: "rgba(0,0,0,0)",
    lineWidth: 1,
  });
  const reference = history.length ? history : median;
  refSeries.setData(reference);

  if (Number.isFinite(currentPrice) && currentPrice > 0) {
    refSeries.createPriceLine({
      price: currentPrice,
      color: "#8fa7cf",
      lineWidth: 1,
      lineStyle: lw.LineStyle.Dashed,
      axisLabelVisible: true,
      title: "Now",
    });
  }

  const targetLines = [
    ["TP1", targets?.tp1, "#00e5a0"],
    ["TP2", targets?.tp2, "#2bd47f"],
    ["TP3", targets?.tp3, "#5ecf73"],
    ["SL1", targets?.sl1, "#ff6b6b"],
    ["SL2", targets?.sl2, "#ff8fa3"],
    ["SL3", targets?.sl3, "#fca5a5"],
  ];
  targetLines.forEach(([label, value, color]) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return;
    refSeries.createPriceLine({
      price: num,
      color,
      lineWidth: 1,
      lineStyle: lw.LineStyle.Dashed,
      axisLabelVisible: true,
      title: label,
    });
  });

  if (row && (!targets || Object.keys(targets).length === 0)) {
    const scenarios = getScenarios(row);
    scenarios.up.tps.slice(0, 3).forEach((val, idx) => {
      refSeries.createPriceLine({
        price: val,
        color: "#00e5a0",
        lineWidth: 1,
        lineStyle: lw.LineStyle.Dashed,
        axisLabelVisible: true,
        title: `TP${idx + 1}`,
      });
    });
    scenarios.up.sls.slice(0, 3).forEach((val, idx) => {
      refSeries.createPriceLine({
        price: val,
        color: "#ff6b6b",
        lineWidth: 1,
        lineStyle: lw.LineStyle.Dashed,
        axisLabelVisible: true,
        title: `SL${idx + 1}`,
      });
    });
  }

  projectionChart.timeScale().fitContent();
  if (typeof ResizeObserver !== "undefined") {
    projectionResizeObserver = new ResizeObserver(() => {
      if (!projectionChart || !detailProjectionChart) return;
      projectionChart.applyOptions({ width: detailProjectionChart.clientWidth });
    });
    projectionResizeObserver.observe(detailProjectionChart);
  }

  const model = data?.model || {};
  const horizon = Number(model.horizon_days);
  const timeframe = model.timeframe ? String(model.timeframe).toUpperCase() : "—";
  const pUpText = formatPct(model.p_up ?? null, 1);
  const confText = Number.isFinite(Number(model.confidence))
    ? formatPct(model.confidence, 1)
    : "—";
  const generated = formatDate(data?.generated_at);
  const horizonText = Number.isFinite(horizon) ? `${horizon}d` : "—";
  renderProjectionState(
    `Projection ${timeframe} ${horizonText} · P(up) ${pUpText} · Confidence ${confText} · ${generated}`,
    targets
  );
}

async function loadProjectionForSymbol(symbol, row = null, scopeOverride = null) {
  if (!detailProjectionChart) return;
  if (!symbol || symbol === "—") {
    resetProjectionChart(true);
    renderProjectionState("Projection pending.");
    return;
  }

  const token = ++projectionReqToken;
  resetProjectionChart();
  renderProjectionState(`Loading projection for ${symbol}...`);

  try {
    const scope = normalizeScope(scopeOverride || resolveScopeForRow(row));
    const mode = scopeToMode(scope);
    let url = withQueryParam(DEFAULT_PROJECTION, "ticker", symbol);
    url = withQueryParam(url, "timeframe", mode);
    const data = await fetchJson(url);
    if (token !== projectionReqToken) return;
    renderProjectionChart(data, row);
  } catch (err) {
    if (token !== projectionReqToken) return;
    resetProjectionChart();
    renderProjectionState(`Projection error: ${err?.message || "Failed to load projection"}`);
  }
}

let overlayChart = null;
let overlayResizeObserver = null;

function renderOverlayChart(symbol, row = null) {
  if (!detailOverlayChart) return;
  const lw = window.LightweightCharts;
  if (!lw) {
    detailOverlayChart.innerHTML = "";
    if (detailOverlayMeta) detailOverlayMeta.textContent = "Overlay chart library not loaded.";
    return;
  }

  if (!overlayData) {
    detailOverlayChart.innerHTML = "";
    if (detailOverlayMeta) {
      const overlayUrl = overlayInput ? overlayInput.value.trim() : "";
      detailOverlayMeta.textContent = overlayUrl ? "No overlay data for this symbol." : "Overlay URL not set.";
    }
    if (detailOverlaySummary) detailOverlaySummary.innerHTML = "";
    return;
  }

  const overlay = getOverlayForSymbol(symbol);
  if (!overlay) {
    detailOverlayChart.innerHTML = "";
    if (detailOverlayMeta) detailOverlayMeta.textContent = "No overlay data for this symbol.";
    if (detailOverlaySummary) detailOverlaySummary.innerHTML = "";
    return;
  }

  const rawSeries = overlay.series || overlay.prices || overlay.data || overlay.bars || [];
  const series = rawSeries
    .map((pt) => {
      const time = toUnixSeconds(pt.time ?? pt.t ?? pt.date ?? pt.timestamp);
      const value = Number(pt.value ?? pt.close ?? pt.price ?? pt.v);
      if (!time || Number.isNaN(value)) return null;
      return { time, value };
    })
    .filter(Boolean);

  if (!series.length) {
    detailOverlayChart.innerHTML = "";
    if (detailOverlayMeta) detailOverlayMeta.textContent = "Overlay data missing series.";
    if (detailOverlaySummary) detailOverlaySummary.innerHTML = "";
    return;
  }

  detailOverlayChart.innerHTML = "";
  if (overlayResizeObserver) overlayResizeObserver.disconnect();
  if (overlayChart) overlayChart = null;

  overlayChart = lw.createChart(detailOverlayChart, {
    layout: {
      background: { type: "solid", color: "#070b15" },
      textColor: "#c5d0e6",
      fontFamily: "Space Grotesk, sans-serif",
    },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.05)" },
      horzLines: { color: "rgba(255,255,255,0.05)" },
    },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.1)" },
    timeScale: { borderColor: "rgba(255,255,255,0.1)" },
    height: detailOverlayChart.clientHeight || 260,
  });

  const zones = Array.isArray(overlay.zones) ? overlay.zones : [];
  zones.forEach((zone) => {
    const from = Number(zone.from);
    const to = Number(zone.to);
    if (!Number.isFinite(from) || !Number.isFinite(to)) return;
    const color = zone.color || "rgba(0,229,160,0.12)";
    const zoneSeries = overlayChart.addBaselineSeries({
      baseValue: { type: "price", price: from },
      topFillColor: color,
      bottomFillColor: color,
      topLineColor: "rgba(0,0,0,0)",
      bottomLineColor: "rgba(0,0,0,0)",
      lineWidth: 1,
    });
    zoneSeries.setData(series.map((pt) => ({ time: pt.time, value: to })));
  });

  const priceSeries = overlayChart.addLineSeries({
    color: "#12d6ff",
    lineWidth: 2,
  });
  priceSeries.setData(series);

  const lines = Array.isArray(overlay.lines) ? overlay.lines : [];
  lines.forEach((line) => {
    if (Array.isArray(line.points) && line.points.length) {
      const lineSeries = overlayChart.addLineSeries({
        color: line.color || "#ffd479",
        lineWidth: line.width || 2,
        lineStyle: resolveLineStyle(line.style),
      });
      const pts = line.points
        .map((pt) => {
          const time = toUnixSeconds(pt.time ?? pt.t ?? pt.date);
          const value = Number(pt.value ?? pt.price);
          if (!time || Number.isNaN(value)) return null;
          return { time, value };
        })
        .filter(Boolean);
      if (pts.length) lineSeries.setData(pts);
    } else if (line.value !== undefined && line.value !== null) {
      priceSeries.createPriceLine({
        price: Number(line.value),
        color: line.color || "#ffd479",
        lineWidth: line.width || 1,
        lineStyle: resolveLineStyle(line.style),
        axisLabelVisible: line.axisLabelVisible !== false,
        title: line.label || "",
      });
    }
  });

  const levels = Array.isArray(overlay.levels) ? overlay.levels : [];
  levels.forEach((lvl) => {
    if (lvl.value === undefined || lvl.value === null) return;
    priceSeries.createPriceLine({
      price: Number(lvl.value),
      color: lvl.color || "#00e5a0",
      lineWidth: lvl.width || 1,
      lineStyle: resolveLineStyle(lvl.style),
      axisLabelVisible: lvl.axisLabelVisible !== false,
      title: lvl.label || "",
    });
  });

  // Scenario lines from row meta (targets / stops)
  if (row) {
    const scenarios = getScenarios(row);
    const addLines = (arr, color, prefix) => {
      arr.forEach((val, idx) => {
        priceSeries.createPriceLine({
          price: val,
          color,
          lineWidth: 1,
          lineStyle: lw.LineStyle.Dashed,
          axisLabelVisible: true,
          title: `${prefix}${idx + 1}`,
        });
      });
    };
    addLines(scenarios.up.tps, "#00e5a0", "TP");
    addLines(scenarios.up.sls, "#ff6b6b", "SL");
    addLines(scenarios.down.tps, "#ffd479", "D-TP");
    addLines(scenarios.down.sls, "#ff8fa3", "D-SL");
  }

  const lastValue = series[series.length - 1]?.value;
  const percentLevels = Array.isArray(overlay.percent_levels) ? overlay.percent_levels : [];
  percentLevels.forEach((pl) => {
    if (!Number.isFinite(lastValue)) return;
    const pct = Number(pl.percent);
    if (Number.isNaN(pct)) return;
    const price = lastValue * (1 + pct);
    const pctLabel = pl.label || `${(pct * 100).toFixed(2)}%`;
    priceSeries.createPriceLine({
      price,
      color: pl.color || "#9aa8bf",
      lineWidth: pl.width || 1,
      lineStyle: resolveLineStyle(pl.style || "dashed"),
      axisLabelVisible: pl.axisLabelVisible !== false,
      title: pctLabel,
    });
  });

  const markers = Array.isArray(overlay.markers) ? overlay.markers : [];
  if (markers.length) {
    const mapped = markers
      .map((m) => {
        const time = toUnixSeconds(m.time ?? m.t ?? m.date);
        if (!time) return null;
        return {
          time,
          position: m.position || "aboveBar",
          color: m.color || "#00e5a0",
          shape: m.shape || "circle",
          text: m.text || "",
        };
      })
      .filter(Boolean);
    if (mapped.length) priceSeries.setMarkers(mapped);
  }

  overlayChart.timeScale().fitContent();
  if (typeof ResizeObserver !== "undefined") {
    overlayResizeObserver = new ResizeObserver(() => {
      overlayChart.applyOptions({ width: detailOverlayChart.clientWidth });
    });
    overlayResizeObserver.observe(detailOverlayChart);
  }

  if (detailOverlayMeta) {
    detailOverlayMeta.textContent = overlay.asof
      ? `Overlay as of ${formatDate(overlay.asof)}`
      : "Overlay loaded.";
  }

  if (detailOverlaySummary) {
    const summaryRows = [];
    summaryRows.push(`<div class="summary-row"><span>Series points</span><span>${series.length}</span></div>`);
    if (Number.isFinite(lastValue)) {
      summaryRows.push(
        `<div class="summary-row"><span>Last value</span><span>${Number(lastValue).toFixed(2)}</span></div>`
      );
    }
    if (Array.isArray(overlay.zones) && overlay.zones.length) {
      summaryRows.push(
        `<div class="summary-row"><span>Zones</span><span>${overlay.zones.length}</span></div>`
      );
    }
    if (Array.isArray(overlay.lines) && overlay.lines.length) {
      summaryRows.push(
        `<div class="summary-row"><span>Lines</span><span>${overlay.lines.length}</span></div>`
      );
    }
    if (Array.isArray(overlay.levels) && overlay.levels.length) {
      summaryRows.push(
        `<div class="summary-row"><span>Levels</span><span>${overlay.levels.length}</span></div>`
      );
    }
    if (Array.isArray(overlay.percent_levels) && overlay.percent_levels.length) {
      summaryRows.push(
        `<div class="summary-row"><span>Percent levels</span><span>${overlay.percent_levels.length}</span></div>`
      );
    }
    if (Array.isArray(overlay.markers) && overlay.markers.length) {
      summaryRows.push(
        `<div class="summary-row"><span>Markers</span><span>${overlay.markers.length}</span></div>`
      );
    }
    detailOverlaySummary.innerHTML = summaryRows.join("");
  }
}

function renderMiniOverlayChart(container, symbol, row = null) {
  if (!container) return;
  const lw = window.LightweightCharts;
  container.innerHTML = "";
  if (!lw) {
    container.innerHTML = `<div class="small-note">Chart lib not loaded.</div>`;
    return;
  }
  const overlay = getOverlayForSymbol(symbol);
  if (!overlay) {
    container.innerHTML = `<div class="small-note">No overlay for ${escapeHtml(symbol)}</div>`;
    return;
  }
  const rawSeries = overlay.series || overlay.prices || overlay.data || overlay.bars || [];
  const series = rawSeries
    .map((pt) => {
      const time = toUnixSeconds(pt.time ?? pt.t ?? pt.date ?? pt.timestamp);
      const value = Number(pt.value ?? pt.close ?? pt.price ?? pt.v);
      if (!time || Number.isNaN(value)) return null;
      return { time, value };
    })
    .filter(Boolean)
    .slice(-150);
  if (!series.length) {
    container.innerHTML = `<div class="small-note">Overlay missing prices.</div>`;
    return;
  }
  const chart = lw.createChart(container, {
    layout: {
      background: { type: "solid", color: "rgba(7,11,21,0.65)" },
      textColor: "#c5d0e6",
      fontFamily: "Space Grotesk, sans-serif",
    },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.05)" },
      horzLines: { color: "rgba(255,255,255,0.05)" },
    },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.05)" },
    timeScale: { borderColor: "rgba(255,255,255,0.05)" },
    height: container.clientHeight || 120,
  });
  const line = chart.addLineSeries({ color: "#12d6ff", lineWidth: 2 });
  line.setData(series);
  const last = series[series.length - 1]?.value;
  if (Array.isArray(overlay.lines)) {
    overlay.lines.slice(0, 2).forEach((lvl) => {
      if (lvl.value === undefined) return;
      line.createPriceLine({
        price: Number(lvl.value),
        color: lvl.color || "#ffd479",
        lineStyle: resolveLineStyle(lvl.style),
        lineWidth: lvl.width || 1,
        title: lvl.label || "",
      });
    });
  }
  if (Array.isArray(overlay.percent_levels) && last) {
    overlay.percent_levels.slice(0, 1).forEach((pl) => {
      const pct = Number(pl.percent);
      if (Number.isNaN(pct)) return;
      line.createPriceLine({
        price: last * (1 + pct),
        color: pl.color || "#9aa8bf",
        lineStyle: resolveLineStyle(pl.style || "dashed"),
        lineWidth: pl.width || 1,
        title: pl.label || `${(pct * 100).toFixed(1)}%`,
      });
    });
  }
  if (row) {
    const scenarios = getScenarios(row);
    const addLines = (arr, color, prefix) => {
      arr.slice(0, 3).forEach((val, idx) => {
        line.createPriceLine({
          price: val,
          color,
          lineStyle: lw.LineStyle.Dashed,
          lineWidth: 1,
          title: `${prefix}${idx + 1}`,
        });
      });
    };
    addLines(scenarios.up.tps, "#00e5a0", "TP");
    addLines(scenarios.up.sls, "#ff6b6b", "SL");
  }
  chart.timeScale().fitContent();
}

function scoreBar(label, value) {
  const pct = Math.max(0, Math.min(1, value || 0));
  return `
    <div>
      <div class="watch-meta"><span>${escapeHtml(label)}</span><span>${(pct * 100).toFixed(1)}%</span></div>
      <div class="score-bar"><span style="width:${pct * 100}%"></span></div>
    </div>
  `;
}

function renderScoreBars(rows, sortSpec) {
  if (!rows || rows.length === 0) {
    scoreBars.innerHTML = `<div class="small-note">No ranked scores available yet.</div>`;
    return;
  }
  const spec = getWatchlistSortSpec(sortSpec?.mode || sortSpec || watchlistSort?.value || "score_desc");
  const topRows = rows.slice(0, Math.min(12, rows.length));
  scoreBars.innerHTML = topRows
    .map((row, i) => {
      const label = String(row?.ticker || row?.symbol || `Top ${i + 1}`);
      const value = Number(row?.score ?? row?.p_accept ?? 0);
      return scoreBar(label, value);
    })
    .join("");
  if (watchlistMeta && topRows.length > 0) {
    watchlistMeta.dataset.sortMetric = spec.key;
  }
}

function buildWeightsHtml(weights, filter = "top") {
  if (!weights || Object.keys(weights).length === 0) {
    return `<div class="small-note">No weights available</div>`;
  }
  const entries = Object.entries(weights).map(([k, v]) => [k, Number(v)]);
  let filtered = entries;
  if (filter === "pos") {
    filtered = entries.filter(([, v]) => v > 0);
  } else if (filter === "neg") {
    filtered = entries.filter(([, v]) => v < 0);
  }
  const sorted =
    filter === "top"
      ? [...filtered].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      : [...filtered].sort((a, b) => Number(b[1]) - Number(a[1]));
  return sorted
    .slice(0, 8)
    .map(
      ([k, v]) =>
        `<div class="weight-item"><span>${escapeHtml(k)}</span><span>${(Number(v) * 100).toFixed(1)}%</span></div>`
    )
    .join("");
}

function buildWeightStats(weights) {
  if (!weights || Object.keys(weights).length === 0) return "";
  const entries = Object.entries(weights).map(([k, v]) => [k, Number(v)]);
  const positives = entries.filter(([, v]) => v > 0);
  const negatives = entries.filter(([, v]) => v < 0);
  const topPos = positives.sort((a, b) => b[1] - a[1])[0];
  const topNeg = negatives.sort((a, b) => a[1] - b[1])[0];
  const sumPos = positives.reduce((acc, [, v]) => acc + v, 0);
  const sumNeg = negatives.reduce((acc, [, v]) => acc + v, 0);
  const total = sumPos + sumNeg;
  const rows = [];
  rows.push(`<div class="stat-row"><span>Pos / Neg</span><span>${positives.length} / ${negatives.length}</span></div>`);
  rows.push(`<div class="stat-row"><span>Net weight</span><span>${(total * 100).toFixed(1)}%</span></div>`);
  if (topPos) {
    rows.push(
      `<div class="stat-row"><span>Top +</span><span>${escapeHtml(topPos[0])} ${(topPos[1] * 100).toFixed(1)}%</span></div>`
    );
  }
  if (topNeg) {
    rows.push(
      `<div class="stat-row"><span>Top -</span><span>${escapeHtml(topNeg[0])} ${(topNeg[1] * 100).toFixed(1)}%</span></div>`
    );
  }
  return rows.join("");
}

function renderTradingView(symbol, scope = "all") {
  if (!tvChart) return;
  const interval = tradingViewIntervalForScope(scopeToMode(normalizeScope(scope)));
  tvChart.innerHTML = `
    <iframe
      src="https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&theme=dark&style=1&locale=en&toolbarbg=%23070b15&hide_side_toolbar=1&withdateranges=1&allow_symbol_change=0"
      width="100%"
      height="340"
      frameborder="0"
      allowtransparency="true"
      loading="lazy"
    ></iframe>
  `;
}

function renderDetailPanel(row) {
  if (!row) return;
  currentDetailRow = row;
  const symbol = row.ticker || row.symbol || "—";
  const scope = resolveScopeForRow(row);
  const chartInterval = tradingViewIntervalForScope(scope);
  const score = Number(row.score || 0);
  const prob = Number(row.p_accept || 0);
  const scenarios = getScenarios(row);

  if (detailSymbol) detailSymbol.textContent = symbol;
  if (detailScore) {
    detailScore.textContent = `${(score * 100).toFixed(1)}% score`;
    detailScore.classList.toggle("score-pulse", score >= 0.65);
  }
  if (detailProb) {
    const pct = Math.max(0, Math.min(100, prob * 100));
    detailProb.innerHTML = `
      <div class="detail-value gauge">
        <span>${pct.toFixed(1)}% accept</span>
        <div class="prob-meter"><span style="width:${pct}%"></span></div>
      </div>
    `;
  }
  if (detailLabel) detailLabel.textContent = row.label || "—";
  const wFilter = weightsFilter ? weightsFilter.value : "top";
  const weights = row.weights || row.weights_json || {};
  if (detailWeights) detailWeights.innerHTML = buildWeightsHtml(weights, wFilter);
  if (detailWeightStats) detailWeightStats.innerHTML = buildWeightStats(weights);
  const detailCard = detailWeights?.closest(".detail-card");
  if (detailCard) {
    detailCard.classList.toggle("highlight", prob >= 0.7);
  }
  if (engineGrid) {
    const meta = row.meta || row.meta_json || {};
    const engineRows = [];
    const pQlib = meta.p_qlib;
    const pSent = meta.p_news ?? meta.p_social;
    if (pQlib !== undefined || pSent !== undefined) {
      engineRows.push(
        `<div class="engine-item"><span>QLIB vs Sentiment</span><span>${formatPct(pQlib || 0)} / ${formatPct(pSent || 0)}</span></div>`
      );
    }
    const keys = [
      ["p_base", "Base"],
      ["p_ta", "TA"],
      ["p_news", "News"],
      ["p_social", "Social"],
      ["p_mc", "MC"],
      ["p_mc_bar", "MC Bar"],
      ["p_lopez", "Lopez"],
      ["p_options", "Options"],
      ["p_finviz", "Finviz"],
      ["p_direction", "Direction"],
      ["p_regime", "Regime"],
    ];
    keys.forEach(([key, label]) => {
      const val = meta[key];
      if (val === null || val === undefined) return;
      engineRows.push(
        `<div class="engine-item"><span>${label}</span><span>${(Number(val) * 100).toFixed(1)}%</span></div>`
      );
    });
    engineGrid.innerHTML = engineRows.length ? engineRows.join("") : `<div class="small-note">No engine breakdown</div>`;
  }
  if (detailTvBtn) {
    const tvSymbol = encodeURIComponent(symbol);
    detailTvBtn.href = `https://www.tradingview.com/chart/?symbol=${tvSymbol}`;
  }
  if (detailChart) {
    detailChart.innerHTML = `
      <iframe
        src="https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(chartInterval)}&theme=dark&style=1&locale=en&toolbarbg=%23070b15&hide_side_toolbar=1&withdateranges=1&allow_symbol_change=0"
        width="100%"
        height="340"
        frameborder="0"
        allowtransparency="true"
      loading="lazy"
    ></iframe>
  `;
  }
  if (scenarioCard) {
    scenarioCard.innerHTML = renderScenarioCard(row, scenarios);
  }
  if (lastNewsData) renderNews(lastNewsData);
  loadProjectionForSymbol(symbol, row, scope);
  loadOverlayForSymbol(symbol, row, scope);
}

function openSymbolModal(row) {
  const symbol = row.ticker || row.symbol || "—";
  const score = Number(row.score || 0);
  const prob = Number(row.p_accept || 0);

  renderDetailPanel(row);
  modalSymbol.textContent = symbol;
  modalScore.textContent = `${(score * 100).toFixed(1)}% score`;
  modalProb.textContent = `${(prob * 100).toFixed(1)}% accept`;
  modalLabel.textContent = row.label || "—";
  modalWeights.innerHTML = buildWeightsHtml(row.weights || row.weights_json || {});
  renderTradingView(symbol, resolveScopeForRow(row));

  if (window.UIkit) {
    UIkit.modal(modal).show();
  } else {
    modal.classList.add("open");
  }
}

function closeModalFallback() {
  if (!modal) return;
  modal.classList.remove("open");
}

if (modalClose) {
  modalClose.addEventListener("click", closeModalFallback);
}
if (modal) {
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModalFallback();
  });
}

if (detailCopyBtn) {
  detailCopyBtn.addEventListener("click", async () => {
    const symbol = detailSymbol?.textContent?.trim();
    if (!symbol || symbol === "—") return;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(symbol);
      } else {
        const area = document.createElement("textarea");
        area.value = symbol;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        document.body.removeChild(area);
      }
      const original = detailCopyBtn.textContent;
      detailCopyBtn.textContent = "Copied";
      setTimeout(() => {
        detailCopyBtn.textContent = original || "Copy";
      }, 1200);
    } catch (err) {
      // ignore copy failure
    }
  });
}

function renderWatchlist(data) {
  const topN = Number(topNInput.value || 10);
  watchlistGrid.innerHTML = "";
  if (watchlistTable) watchlistTable.innerHTML = "";
  if (!data) {
    watchlistMeta.textContent = "No watchlist data.";
    return;
  }

  lastWatchlistData = data;
  if (denseCardsToggle && watchlistGrid) {
    watchlistGrid.classList.toggle("dense", denseCardsToggle.checked);
  }
  if (compactWeightsToggle && watchlistGrid) {
    watchlistGrid.classList.toggle("compact-weights", compactWeightsToggle.checked);
  }

  const asof = data.asof || data.generated_at || "";
  asofValue.textContent = formatDate(asof);
  sourceValue.textContent = data.source || data.provider || "Supabase artifacts";

  const filterTerm = (watchlistFilter?.value || "").trim().toUpperCase();
  const sortSpec = getWatchlistSortSpec(watchlistSort?.value || "score_desc");
  const capMode = (marketCapFilter?.value || "all").toLowerCase();
  const assetMode = (assetTypeFilter?.value || "all").toLowerCase();

  const activeFilters = [];
  if (filterTerm) activeFilters.push(`text=${filterTerm}`);
  if (capMode !== "all") activeFilters.push(`cap=${capMode}`);
  if (assetMode !== "all") activeFilters.push(`asset=${assetMode}`);
  const filterSuffix = activeFilters.length ? ` · filters ${activeFilters.join(", ")}` : "";

  const rankedList = data.ranked || data.rows || [];
  if (Array.isArray(rankedList)) {
    if (rankedList.length === 0) {
      countValue.textContent = 0;
      watchlistMeta.textContent = data.message || "Ranked list · no rows for this timeframe.";
      renderScoreBars([]);
      clearDetailPanel();
      renderWatchlistTable([]);
      if (lastNewsData) renderNews(lastNewsData);
      return;
    }

    const base = rankedList;
    const filtered = base.filter((row) => {
      const ticker = String(row.ticker || row.symbol || "").toUpperCase();
      const label = String(row.label || "").toUpperCase();
      const plan = String(row.plan_type || "").toUpperCase();
      const textMatch = !filterTerm || ticker.includes(filterTerm) || label.includes(filterTerm) || plan.includes(filterTerm);
      return textMatch && capFilterMatch(row, capMode) && assetFilterMatch(row, assetMode);
    });

    const sorted = [...filtered].sort((a, b) => compareWatchlistRows(a, b, sortSpec));

    const ranked = sorted.slice(0, topN);
    countValue.textContent = rankedList.length;
    watchlistMeta.textContent = `Ranked list · showing ${ranked.length}${filtered.length !== rankedList.length ? " (filtered)" : ""}${filterSuffix} · sort ${describeWatchlistSort(sortSpec)}`;
    renderScoreBars(sorted, sortSpec);

    if (ranked.length) {
      const selectedSymbol = String(currentDetailRow?.ticker || currentDetailRow?.symbol || "").toUpperCase();
      const selectedRow =
        ranked.find((row) => String(row.ticker || row.symbol || "").toUpperCase() === selectedSymbol) || ranked[0];
      renderDetailPanel(selectedRow);
      setActiveCard(selectedRow.ticker || selectedRow.symbol || "—");
    } else {
      clearDetailPanel();
    }

    ranked.forEach((row) => {
      const symbolRaw = row.ticker || row.symbol || "—";
      const symbol = escapeHtml(row.ticker || row.symbol || "—");
      const label = escapeHtml(row.label || "—");
      const plan = escapeHtml(row.plan_type || "—");
      const projectedExit = escapeHtml(getProjectedExitDate(row));
      const capBucket = getRowCapBucket(row);
      const capValue = getRowMarketCap(row);
      const capTxt = `${formatCapBucket(capBucket)} · ${formatMarketCap(capValue)}`;
      const assetTxt = getRowAssetType(row).toUpperCase();
      const card = document.createElement("div");
      card.className = "watch-card";
      card.dataset.symbol = symbolRaw;
      const signalClass = (row.signal || "").toLowerCase();
      card.innerHTML = `
        <div class="watch-head">
          <h4>${symbol}</h4>
          <span class="badge badge-${signalClass}">${label}</span>
        </div>
        <div class="watch-meta"><span>Score</span><span>${(Number(row.score || 0) * 100).toFixed(1)}%</span></div>
        <div class="watch-meta"><span>P(accept)</span><span>${(Number(row.p_accept || 0) * 100).toFixed(1)}%</span></div>
        <div class="watch-meta"><span>Cap</span><span>${escapeHtml(capTxt)}</span></div>
        <div class="watch-meta"><span>Asset</span><span>${escapeHtml(assetTxt)}</span></div>
        <div class="watch-meta"><span>Timeframe</span><span class="badge badge-tf">${plan}</span></div>
        <div class="watch-meta"><span>Projected Exit</span><span>${projectedExit}</span></div>
        <div class="weight-list">${buildWeightsHtml(row.weights || row.weights_json || {})}</div>
      `;
      card.addEventListener("click", () => {
        renderDetailPanel(row);
        setActiveCard(symbolRaw);
        if (lastNewsData) renderNews(lastNewsData);
      });
      card.addEventListener("dblclick", () => {
        openSymbolModal(row);
      });
      watchlistGrid.appendChild(card);
    });

    if (ranked.length && !currentDetailRow) {
      setActiveCard(ranked[0].ticker || ranked[0].symbol || "—");
    }
    renderWatchlistTable(ranked);
    return;
  }

  if (Array.isArray(data.tickers)) {
    const baseRows = data.tickers.map((t) => ({ ticker: t, label: "Universe", p_accept: 0, score: 0, weights: {} }));
    const filteredRows = baseRows.filter((row) => {
      const ticker = String(row.ticker || "").toUpperCase();
      const textMatch = !filterTerm || ticker.includes(filterTerm);
      return textMatch && capFilterMatch(row, capMode) && assetFilterMatch(row, assetMode);
    });

    const rowsSorted = [...filteredRows].sort((a, b) => compareWatchlistRows(a, b, sortSpec));
    const rows = rowsSorted.slice(0, topN);

    countValue.textContent = data.count || data.tickers.length;
    watchlistMeta.textContent = `Universe list · showing ${rows.length}${rows.length !== baseRows.length ? " (filtered)" : ""}${filterSuffix} · sort ${describeWatchlistSort(sortSpec)}. No probabilities in this file.`;
    renderScoreBars(rowsSorted, sortSpec);

    if (rows.length) {
      renderDetailPanel(rows[0]);
    } else {
      clearDetailPanel();
    }

    rows.forEach((row) => {
      const ticker = escapeHtml(row.ticker || "—");
      const card = document.createElement("div");
      card.className = "watch-card";
      card.dataset.symbol = row.ticker || "";
      card.innerHTML = `
        <h4>${ticker}</h4>
        <div class="watch-meta"><span>Universe</span><span>Member</span></div>
        <div class="watch-meta"><span>Cap</span><span>${formatCapBucket(getRowCapBucket(row))}</span></div>
        <div class="watch-meta"><span>Asset</span><span>${getRowAssetType(row).toUpperCase()}</span></div>
        <div class="small-note">No ranking data in this list. Use a ranked watchlist JSON.</div>
      `;
      card.addEventListener("click", () => {
        renderDetailPanel(row);
        setActiveCard(row.ticker || "");
      });
      card.addEventListener("dblclick", () => {
        openSymbolModal(row);
      });
      watchlistGrid.appendChild(card);
    });

    if (rows.length) setActiveCard(rows[0].ticker || "");
    renderWatchlistTable(rows);
    return;
  }

  watchlistMeta.textContent = "Unsupported watchlist format.";
}
function renderWatchlistTable(rows) {
  if (!watchlistTable) return;
  if (!Array.isArray(rows) || rows.length === 0) {
    watchlistTable.innerHTML = "";
    return;
  }
  const currentSort = getWatchlistSortSpec(watchlistSort?.value || "score_desc");
  const sortIndicator = (key) => {
    if (currentSort.key !== key) return "";
    return currentSort.dir === "asc" ? " ↑" : " ↓";
  };

  const rowBySymbol = new Map();
  rows.forEach((row) => {
    const key = String(row.ticker || row.symbol || "").toUpperCase();
    if (key && !rowBySymbol.has(key)) rowBySymbol.set(key, row);
  });

  const renderDetail = (row) => {
    const score = Number(row.score || 0);
    const accept = Number(row.p_accept || 0);
    const hit = getHitProbability(row);
    const scenarios = getScenarios(row);
    const target = getTargetPrice(row) || scenarios.up.tps[0] || null;
    const weights = row.weights || row.weights_json || {};
    const meta = getMeta(row);
    const projectedExit = getProjectedExitDate(row);
    const capValue = getRowMarketCap(row);
    return `
      <div class="row-detail">
        <div class="prob-stack">
          <div class="prob-chip"><span>Score</span><span>${(score * 100).toFixed(1)}%</span></div>
          <div class="prob-chip"><span>P(accept)</span><span>${(accept * 100).toFixed(1)}%</span></div>
          <div class="prob-chip"><span>P(hit target)</span><span>${(hit * 100).toFixed(1)}%</span></div>
          <div class="prob-chip"><span>Target</span><span>${target ? `$${Number(target).toFixed(2)}` : "—"}</span></div>
        </div>
        <div class="detail-flex">
          <div class="mini-block">
            <div class="mini-label">Overlay</div>
            <div class="mini-chart" data-symbol="${escapeHtml(row.ticker || row.symbol || "")}"></div>
          </div>
          <div class="mini-block">
            <div class="mini-label">Weights</div>
            <div class="mini-weights">${buildWeightsHtml(weights, "top")}</div>
          </div>
          <div class="mini-block">
            <div class="mini-label">Meta</div>
            <div class="mini-meta">
              <div class="stat-row"><span>Plan</span><span>${escapeHtml(row.plan_type || "—")}</span></div>
              <div class="stat-row"><span>Projected Exit</span><span>${escapeHtml(projectedExit)}</span></div>
              <div class="stat-row"><span>Cap</span><span>${escapeHtml(`${formatCapBucket(getRowCapBucket(row))} · ${formatMarketCap(capValue)}`)}</span></div>
              <div class="stat-row"><span>Asset</span><span>${escapeHtml(getRowAssetType(row).toUpperCase())}</span></div>
              ${meta.p_base !== undefined ? `<div class="stat-row"><span>Base</span><span>${formatPct(meta.p_base)}</span></div>` : ""}
              ${meta.p_direction !== undefined ? `<div class="stat-row"><span>Direction</span><span>${formatPct(meta.p_direction)}</span></div>` : ""}
            </div>
          </div>
        </div>
      </div>
    `;
  };

  const body = rows
    .map((row) => {
      const symbolRaw = String(row.ticker || row.symbol || "");
      const symbolKey = symbolRaw.toUpperCase();
      const symbol = escapeHtml(symbolRaw || "—");
      const plan = escapeHtml(row.plan_type || "—");
      const score = `${(Number(row.score || 0) * 100).toFixed(1)}%`;
      const accept = `${(Number(row.p_accept || 0) * 100).toFixed(1)}%`;
      const hit = `${(getHitProbability(row) * 100).toFixed(1)}%`;
      const scenarios = getScenarios(row);
      const target = getTargetPrice(row) || scenarios.up.tps[0] || null;
      const targetTxt = target ? `$${Number(target).toFixed(2)}` : "—";
      const expRetRaw =
        row?.expected_return_pct !== undefined && row?.expected_return_pct !== null
          ? Number(row.expected_return_pct)
          : Number(row?.expected_return ?? 0) * 100;
      const expRetTxt = Number.isFinite(expRetRaw) ? `${expRetRaw.toFixed(2)}%` : "—";
      const capTxt = `${formatCapBucket(getRowCapBucket(row))} · ${formatMarketCap(getRowMarketCap(row))}`;
      const assetTxt = getRowAssetType(row).toUpperCase();
      return `
        <tr class="main-row" data-symbol="${escapeHtml(symbolKey)}" tabindex="0" role="button" title="Click to expand details. Double-click to open modal.">
          <td><span class="caret">&#9656;</span>${symbol}</td>
          <td>${score}</td>
          <td>${accept}</td>
          <td>${hit}</td>
          <td>${expRetTxt}</td>
          <td>${targetTxt}</td>
          <td>${escapeHtml(capTxt)}</td>
          <td>${escapeHtml(assetTxt)}</td>
          <td>${plan}</td>
        </tr>
        <tr class="detail-row hidden" data-symbol="${escapeHtml(symbolKey)}">
          <td colspan="9">${renderDetail(row)}</td>
        </tr>
      `;
    })
    .join("");

  watchlistTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th class="sortable" data-sort="ticker">Ticker${sortIndicator("ticker")}</th>
          <th class="sortable" data-sort="score">Score${sortIndicator("score")}</th>
          <th class="sortable" data-sort="accept">P(accept)${sortIndicator("accept")}</th>
          <th>P(hit target)</th>
          <th class="sortable" data-sort="expected_return">Exp Return${sortIndicator("expected_return")}</th>
          <th>Target</th>
          <th class="sortable" data-sort="market_cap">Cap${sortIndicator("market_cap")}</th>
          <th>Asset</th>
          <th class="sortable" data-sort="horizon">Plan${sortIndicator("horizon")}</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;

  const activateRow = (tr) => {
    const symbol = String(tr?.dataset?.symbol || "").toUpperCase();
    if (!symbol) return;
    const row = rowBySymbol.get(symbol);
    const detailRow = tr.nextElementSibling;
    const caret = tr.querySelector(".caret");

    if (detailRow && detailRow.classList.contains("detail-row")) {
      const isHidden = detailRow.classList.toggle("hidden");
      if (caret) caret.textContent = isHidden ? "\u25B8" : "\u25BE";
      if (!detailRow.dataset.rendered && !isHidden) {
        const mini = detailRow.querySelector(".mini-chart");
        if (mini) renderMiniOverlayChart(mini, symbol, row);
        detailRow.dataset.rendered = "1";
      }
    }

    if (row) {
      renderDetailPanel(row);
      setActiveCard(symbol);
      if (lastNewsData) renderNews(lastNewsData);
    }
  };

  watchlistTable.querySelectorAll("tbody tr.main-row").forEach((tr) => {
    tr.addEventListener("click", () => activateRow(tr));
    tr.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        activateRow(tr);
      }
    });
    tr.addEventListener("dblclick", () => {
      const symbol = String(tr.dataset.symbol || "").toUpperCase();
      const row = rowBySymbol.get(symbol);
      if (row) openSymbolModal(row);
    });
  });

  watchlistTable.querySelectorAll("thead th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = String(th.dataset.sort || "").trim();
      if (!key) return;
      const current = getWatchlistSortSpec(watchlistSort?.value || "score_desc");
      let nextDir = "desc";
      if (current.key === key) {
        nextDir = current.dir === "desc" ? "asc" : "desc";
      } else if (key === "ticker") {
        nextDir = "asc";
      }
      setWatchlistSort(`${key}_${nextDir}`, { persist: true, rerender: true });
    });
  });
}
function clearDetailPanel() {
  if (detailSymbol) detailSymbol.textContent = "—";
  if (detailScore) detailScore.textContent = "—";
  if (detailProb) detailProb.textContent = "—";
  if (detailLabel) detailLabel.textContent = "—";
  if (detailWeights) detailWeights.innerHTML = "";
  if (detailChart) detailChart.innerHTML = "";
  resetProjectionChart(true);
  if (detailProjectionMeta) detailProjectionMeta.textContent = "Projection pending.";
  if (detailProjectionTargets) detailProjectionTargets.innerHTML = "";
  if (detailOverlayChart) detailOverlayChart.innerHTML = "";
  if (detailOverlayMeta) detailOverlayMeta.textContent = "Overlay data pending.";
  if (detailOverlaySummary) detailOverlaySummary.innerHTML = "";
  if (scenarioCard) scenarioCard.innerHTML = "";
}

function setActiveCard(symbol) {
  if (!symbol || !watchlistGrid) return;
  const target = String(symbol).toUpperCase();
  const cards = watchlistGrid.querySelectorAll(".watch-card");
  cards.forEach((card) => {
    const sym = (card.dataset.symbol || "").toUpperCase();
    if (sym === target) {
      card.classList.add("active");
    } else {
      card.classList.remove("active");
    }
  });
}

function renderNews(data) {
  if (!newsGrid) return;
  newsGrid.innerHTML = "";
  lastNewsData = data;

  if (!data) {
    newsMeta.textContent = "No news data.";
    return;
  }

  const items = normalizeNewsItems(data);
  if (!items.length) {
    newsMeta.textContent = "No news rows available.";
    return;
  }

  const selectedSymbol = String(currentDetailRow?.ticker || currentDetailRow?.symbol || "").toUpperCase();
  const watchlistRows = lastWatchlistData?.ranked || lastWatchlistData?.rows || [];
  const watchlistSymbols = new Set(
    watchlistRows
      .map((row) => String(row?.ticker || row?.symbol || "").toUpperCase())
      .filter(Boolean)
  );

  const prioritized = items
    .map((item) => {
      const matchesSelected = selectedSymbol ? item.tickers.includes(selectedSymbol) : false;
      const matchesWatchlist = item.tickers.some((t) => watchlistSymbols.has(t));
      return { ...item, matchesSelected, matchesWatchlist };
    })
    .sort((a, b) => {
      if (a.matchesSelected !== b.matchesSelected) return a.matchesSelected ? -1 : 1;
      if (a.matchesWatchlist !== b.matchesWatchlist) return a.matchesWatchlist ? -1 : 1;
      const ta = new Date(a.published || 0).getTime();
      const tb = new Date(b.published || 0).getTime();
      return (Number.isFinite(tb) ? tb : 0) - (Number.isFinite(ta) ? ta : 0);
    });

  const top = prioritized.slice(0, 12);
  const selectedCount = selectedSymbol ? prioritized.filter((n) => n.matchesSelected).length : 0;
  const watchlistCount = prioritized.filter((n) => n.matchesWatchlist).length;

  if (selectedSymbol) {
    newsMeta.textContent = `Showing ${top.length} of ${items.length} · ${selectedCount} mention ${selectedSymbol}`;
  } else {
    newsMeta.textContent = `Showing ${top.length} of ${items.length} · ${watchlistCount} tied to watchlist symbols`;
  }

  top.forEach((item) => {
    const title = escapeHtml(item.title || "Untitled");
    const url = item.url || "#";
    const ts = item.published;
    const tickersDisplay = item.tickers.length ? escapeHtml(item.tickers.slice(0, 4).join(", ")) : "—";
    const sentimentText = item.sentiment === null ? "—" : Number(item.sentiment).toFixed(2);

    const domainLabel = (() => {
      if (!url || url === "#") return "";
      try {
        const u = new URL(url);
        return u.hostname.replace("www.", "");
      } catch (err) {
        return "";
      }
    })();

    const sourceLabel = escapeHtml(item.source || domainLabel || "Open source");
    const card = document.createElement("div");
    card.className = `news-card${item.matchesSelected ? " focus" : ""}${url !== "#" ? " clickable" : ""}`;
    card.innerHTML = `
      <h5>${title}</h5>
      <div class="news-meta">
        <span>${formatDate(ts)}</span>
        <span>${tickersDisplay}</span>
      </div>
      <div class="news-meta">
        <span>Sentiment</span>
        <span>${sentimentText}</span>
      </div>
      <div class="small-note"><a href="${url}" target="_blank" rel="noreferrer noopener">${sourceLabel}</a></div>
    `;

    if (url !== "#") {
      card.tabIndex = 0;
      card.setAttribute("role", "link");
      card.addEventListener("click", (ev) => {
        if (ev.target && ev.target.closest("a")) return;
        window.open(url, "_blank", "noopener,noreferrer");
      });
      card.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          window.open(url, "_blank", "noopener,noreferrer");
        }
      });
    }

    newsGrid.appendChild(card);
  });
}
function renderWalkforward(data) {
  if (!walkforwardGrid) return;
  walkforwardGrid.innerHTML = "";
  if (!data) {
    if (walkforwardMeta) walkforwardMeta.textContent = "No walk-forward data.";
    return;
  }
  // Support either {summary:{...}} or raw summary object
  const summary = data.summary || data;
  const stats = summary.stats || {};
  const topWeightsRaw = summary.weights_top || summary.weights || [];
  const topWeights = Array.isArray(topWeightsRaw)
    ? topWeightsRaw
    : Object.entries(topWeightsRaw).map(([rule, weight]) => ({ rule, weight }));
  const scope = String(summary.timeframe || "all");
  const diagnostics = summary.diagnostics || {};
  const probabilityDiag = diagnostics.probability || {};
  const concentrationDiag = diagnostics.concentration || {};
  const driftDiag = diagnostics.temporal_drift || null;
  const benchmarkDiag = diagnostics.benchmarks || {};
  const oosDelta = diagnostics.oos_delta || {};
  const rollingWindows = Array.isArray(diagnostics.rolling_windows) ? diagnostics.rolling_windows : [];
  const capBuckets = Array.isArray(diagnostics.cap_bucket_stability) ? diagnostics.cap_bucket_stability : [];
  const toolRegistry = Array.isArray(diagnostics.tool_registry) ? diagnostics.tool_registry : [];
  const methodCounts = diagnostics.coverage?.method_counts || {};
  const fmtNum = (value, digits = 3) => {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "—";
  };
  const fmtPctMaybe = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? formatPct(n) : "—";
  };
  const fmtToolWeight = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return Math.abs(n) <= 1 ? formatPct(n) : n.toFixed(3);
  };

  const cards = [];
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">As of</div>
      <div class="wf-value">${escapeHtml(formatDate(summary.asof || ""))}</div>
      <div class="wf-small">Scope ${escapeHtml(scope)} · Horizon ${escapeHtml(String(summary.horizon || "?"))}d · Top rules ${escapeHtml(String(summary.top_rules || "?"))}</div>
    </div>
  `);
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Rule Counts</div>
      <div class="wf-value">${escapeHtml(String(stats.total_rules || 0))}</div>
      <div class="wf-small">Pos ${escapeHtml(String(stats.pos_count || 0))} · Neg ${escapeHtml(String(stats.neg_count || 0))}</div>
    </div>
  `);
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Net Weight</div>
      <div class="wf-value">${fmtPctMaybe(stats.net_weight)}</div>
      <div class="wf-small">CI ${fmtPctMaybe(stats.net_weight_ci_low)} to ${fmtPctMaybe(stats.net_weight_ci_high)}</div>
    </div>
  `);

  const topList = topWeights.slice(0, 6)
    .map((t) => `<div class="engine-item"><span>${escapeHtml(t.rule || "")}</span><span>${formatPct(Number(t.weight) || 0)}</span></div>`)
    .join("");
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Top Weights</div>
      ${topList || '<div class="wf-small">No weights</div>'}
    </div>
  `);
  const topTools = toolRegistry
    .slice(0, 8)
    .map((tool) => {
      const status = String(tool?.status || "idle").toUpperCase();
      return `<div class="engine-item"><span>${escapeHtml(String(tool?.name || tool?.key || ""))} <small class="wf-small">${escapeHtml(status)}</small></span><span>${fmtToolWeight(tool?.weight)}</span></div>`;
    })
    .join("");
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Tool Coverage</div>
      ${topTools || '<div class="wf-small">No tool registry in payload.</div>'}
    </div>
  `);
  const methodsText = Object.entries(methodCounts)
    .map(([k, v]) => `${k}:${v}`)
    .join(" · ");
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Probability Quality</div>
      <div class="wf-value">Conf ${fmtPctMaybe(probabilityDiag.avg_confidence ?? stats.avg_confidence)}</div>
      <div class="wf-small">Entropy ${fmtNum(probabilityDiag.avg_entropy ?? stats.avg_entropy, 3)} · P(up) ${fmtPctMaybe(probabilityDiag.avg_p_accept ?? stats.avg_p_accept)}</div>
    </div>
  `);
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Rule Concentration</div>
      <div class="wf-value">Eff ${fmtNum(concentrationDiag.effective_rules, 2)}</div>
      <div class="wf-small">HHI ${fmtNum(concentrationDiag.hhi, 3)} · Top share ${fmtPctMaybe(concentrationDiag.top_rule_share)}</div>
    </div>
  `);
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Temporal Drift</div>
      <div class="wf-value">${fmtPctMaybe(oosDelta.net_weight_delta ?? driftDiag?.delta)}</div>
      <div class="wf-small">Newest ${fmtPctMaybe(driftDiag?.newest_mean)} · Oldest ${fmtPctMaybe(driftDiag?.oldest_mean)}</div>
      <div class="wf-small">Conf Δ ${fmtPctMaybe(oosDelta.confidence_delta)}</div>
      <div class="wf-small">${methodsText ? `Methods ${escapeHtml(methodsText)}` : "Methods unavailable"}</div>
    </div>
  `);
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">Benchmarks</div>
      <div class="wf-value">Entropy edge ${fmtNum(benchmarkDiag.entropy_edge, 3)}</div>
      <div class="wf-small">Conf edge ${fmtNum(benchmarkDiag.confidence_edge_vs_neutral, 3)} · Directional edge ${fmtNum(benchmarkDiag.directional_edge_abs_pdiff, 3)}</div>
      <div class="wf-small">Neutral entropy ${fmtNum(benchmarkDiag.neutral_entropy_baseline, 3)}</div>
    </div>
  `);
  const capTop = capBuckets
    .slice(0, 3)
    .map((c) => `${String(c.bucket || "").toUpperCase()}:${c.rows}`)
    .join(" · ");
  const latestWin = rollingWindows.length ? rollingWindows[rollingWindows.length - 1] : null;
  cards.push(`
    <div class="wf-card">
      <div class="wf-title">OOS Windows</div>
      <div class="wf-value">${rollingWindows.length}</div>
      <div class="wf-small">Latest net ${fmtPctMaybe(latestWin?.avg_net_weight)} · Latest conf ${fmtPctMaybe(latestWin?.avg_confidence)}</div>
      <div class="wf-small">${capTop ? `Cap buckets ${escapeHtml(capTop)}` : "Cap buckets unavailable"}</div>
    </div>
  `);

  walkforwardGrid.innerHTML = cards.join("");
  if (walkforwardMeta) {
    const scopeMeta = summary.timeframe ? ` · scope ${summary.timeframe}` : "";
    walkforwardMeta.textContent = summary.run_id
      ? `Run ${summary.run_id} · rows ${summary.signals_rows || "?"}${scopeMeta}`
      : "Walk-forward summary loaded.";
  }
}

async function fetchJson(url) {
  if (!url) return null;
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`Fetch failed: ${resp.status}`);
  }
  try {
    return await resp.json();
  } catch (err) {
    throw new Error("Failed to parse JSON");
  }
}

function transformApiToWatchlist(apiData) {
  if (!apiData || !apiData.forecasts) return null;
  // Transform /api/forecasts response to watchlist format
  const forecasts = apiData.forecasts;
  const rows = forecasts.map((f) => ({
    ticker: f.ticker,
    score: f.accept || 0,
    p_accept: f.accept || 0,
    p_reject: f.reject || 0,
    p_continue: f.continue || 0,
    confidence: f.confidence || 0,
    method: f.method || "blended",
    meta: {
      p_base: f.accept,
      weights: f.weights || {},
      explain: f.explain || {},
      horizon: f.horizon || {},
      run_id: f.run_id,
    },
    created_at: f.date,
  }));
  return {
    asof: apiData.generated_at || new Date().toISOString(),
    source: "API (live Supabase)",
    count: rows.length,
    rows: rows,
  };
}

function normalizeScope(rawScope) {
  const scope = String(rawScope || "all").toLowerCase();
  if (scope === "day" || scope === "swing" || scope === "long" || scope === "all") return scope;
  return "all";
}

function getSelectedRunId() {
  const fromSelect = String(runSel ? runSel.value : currentRunId || "").trim();
  return fromSelect;
}

function getSelectedScope() {
  return normalizeScope(timeframeSel ? timeframeSel.value : "all");
}

function scopeToMode(scope) {
  return scope === "all" ? "swing" : scope;
}

function resolveScopeForRow(row) {
  const selected = getSelectedScope();
  if (selected !== "all") return selected;
  const plan = normalizeScope(row?.plan_type || "");
  return plan === "all" ? "swing" : plan;
}

function tradingViewIntervalForScope(scope) {
  if (scope === "day") return "60";
  if (scope === "long") return "W";
  return "D";
}

function normalizeTimeframeCounts(rawCounts) {
  const counts = { day: 0, swing: 0, long: 0 };
  if (rawCounts && typeof rawCounts === "object") {
    for (const key of ["day", "swing", "long"]) {
      const raw = Number(rawCounts[key] ?? 0);
      counts[key] = Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 0;
    }
  }
  counts.all = counts.day + counts.swing + counts.long;
  return counts;
}

function syncScopeChips(scope) {
  const chips = document.querySelectorAll(".chip");
  chips.forEach((c) => {
    c.classList.toggle("active", (c.dataset.scope || "all") === scope);
  });
}

function applyTimeframeAvailability(rawCounts) {
  const counts = normalizeTimeframeCounts(rawCounts);
  const chips = document.querySelectorAll(".chip");
  chips.forEach((chip) => {
    const scope = chip.dataset.scope || "all";
    if (scope === "all") {
      chip.disabled = false;
      chip.classList.remove("chip-disabled");
      chip.classList.remove("chip-empty");
      chip.setAttribute("aria-disabled", "false");
      chip.title = `All (${counts.all})`;
      return;
    }
    const unavailable = counts[scope] === 0;
    chip.disabled = unavailable;
    chip.classList.toggle("chip-disabled", unavailable);
    chip.classList.toggle("chip-empty", unavailable);
    chip.setAttribute("aria-disabled", unavailable ? "true" : "false");
    chip.title = unavailable ? `No ${scope} rows in selected run` : `${counts[scope]} ${scope} rows`;
  });

  if (timeframeSel) {
    Array.from(timeframeSel.options).forEach((opt) => {
      const scope = opt.value;
      if (scope === "all") {
        opt.disabled = false;
        return;
      }
      opt.disabled = (counts[scope] || 0) === 0;
    });
  }

  return counts;
}

function summarizeRunLabel(run) {
  if (!run || typeof run !== "object") return "";
  const rid = String(run.run_id || "").trim();
  const shortId = rid ? rid.slice(0, 8) : "n/a";
  const rows = Number(run.rows || 0);
  const c = run.timeframe_counts || {};
  const d = Number(c.day || 0);
  const s = Number(c.swing || 0);
  const l = Number(c.long || 0);
  return `${shortId} · rows ${rows} · d/s/l ${d}/${s}/${l}`;
}

function renderRunCatalog(payload) {
  if (!runSel || !payload || !Array.isArray(payload.runs)) return;
  runCatalog = payload.runs.slice();

  const requested = String(currentRunId || runSel.value || "").trim();
  runSel.innerHTML = "";
  const latestOpt = document.createElement("option");
  latestOpt.value = "";
  latestOpt.textContent = "Latest Run";
  runSel.appendChild(latestOpt);

  runCatalog.forEach((run) => {
    const runId = String(run.run_id || "").trim();
    if (!runId) return;
    const opt = document.createElement("option");
    opt.value = runId;
    opt.textContent = summarizeRunLabel(run);
    runSel.appendChild(opt);
  });

  const hasRequested = requested && runCatalog.some((r) => String(r.run_id || "").trim() === requested);
  currentRunId = hasRequested ? requested : "";
  runSel.value = currentRunId;
  localStorage.setItem("ddl69_run_id", currentRunId);

  const selectedRun = runCatalog.find((r) => String(r.run_id || "").trim() === currentRunId);
  if (runMeta) {
    if (selectedRun) {
      const asof = selectedRun.asof || selectedRun.created_at || "";
      runMeta.textContent = `Run scope: ${String(selectedRun.run_id).slice(0, 12)} · asof ${formatDateShort(asof)} · rows ${selectedRun.rows || 0}`;
    } else {
      runMeta.textContent = `Run scope: latest (${payload.latest_run_id ? String(payload.latest_run_id).slice(0, 12) : "n/a"})`;
    }
  }
}

// ============================================================
// DASHBOARD SECTIONS — Model Perf, Features, MC, Lopez, ML Tools
// ============================================================
const perfMatrixBody = document.getElementById("perfMatrixBody");
const perfMatrixMeta = document.getElementById("perfMatrixMeta");
const featureImportanceBody = document.getElementById("featureImportanceBody");
const featureBadge = document.getElementById("featureBadge");
const mcVaR95 = document.getElementById("mcVaR95");
const mcCVaR95 = document.getElementById("mcCVaR95");
const mcMaxDD = document.getElementById("mcMaxDD");
const mcSharpe = document.getElementById("mcSharpe");
const mcVol = document.getElementById("mcVol");
const mcSims = document.getElementById("mcSims");
const mcBadge = document.getElementById("mcBadge");
const mcMeta = document.getElementById("mcMeta");
const lopezConfidence = document.getElementById("lopezConfidence");
const lopezPAccept = document.getElementById("lopezPAccept");
const lopezReturn = document.getElementById("lopezReturn");
const lopezStrongBuy = document.getElementById("lopezStrongBuy");
const lopezBHP = document.getElementById("lopezBHP");
const lopezTotal = document.getElementById("lopezTotal");
const lopezTableBody = document.getElementById("lopezTableBody");
const mlToolsGrid = document.getElementById("mlToolsGrid");

function renderModelPerformance(auditData) {
  if (!perfMatrixBody) return;
  // TRUTH MODE: perfMatrixMeta is now a static label "Loading audit metrics..." 
  // We do NOT want to overwrite the "Backtest Evaluation" badge in HTML.
  // We only update the text content if it's strictly necessary, or leave it.

  if (!auditData || !auditData.predictions || !auditData.predictions.length) {
    perfMatrixBody.innerHTML = '<tr><td colspan="10" class="loading-placeholder">No audit predictions available</td></tr>';
    if (perfMatrixMeta) perfMatrixMeta.textContent = "No audit data found.";
    return;
  }
  const preds = auditData.predictions.slice(0, 20);
  perfMatrixBody.innerHTML = preds.map((p) => {
    const m = p.metrics || {};
    const convClass = (p.conviction || "").includes("STRONG") ? "ensemble-row" : "";
    const signalClass = p.signal === "BUY" ? "positive" : p.signal === "SELL" ? "negative" : "";
    const confPct = Number.isFinite(m.confidence) ? (m.confidence * 100).toFixed(1) + "%" : "\u2014";
    const pAccPct = Number.isFinite(m.p_accept) ? (m.p_accept * 100).toFixed(1) + "%" : "\u2014";
    const retPct = Number.isFinite(m.expected_return_pct) ? (m.expected_return_pct >= 0 ? "+" : "") + m.expected_return_pct.toFixed(2) + "%" : "\u2014";
    const retClass = Number.isFinite(m.expected_return_pct) ? (m.expected_return_pct >= 0 ? "positive" : "negative") : "";
    const sharpe = Number.isFinite(m.sharpe_estimate) ? m.sharpe_estimate.toFixed(2) : "\u2014";
    const sharpeClass = Number.isFinite(m.sharpe_estimate) && m.sharpe_estimate > 1 ? "positive" : "";
    const rr = Number.isFinite(m.risk_reward_ratio) ? m.risk_reward_ratio.toFixed(2) : "\u2014";
    const horizon = Number.isFinite(m.horizon_days) ? m.horizon_days + "d" : "\u2014";
    return `<tr class="model-row ${convClass}" data-symbol="${escapeHtml(String(p.ticker || ""))}" tabindex="0" role="button" title="Click to focus this ticker in the watchlist.">
      <td class="model-name"><span class="status-dot ${signalClass === "positive" ? "active" : signalClass === "negative" ? "optional" : ""}"></span>${escapeHtml(p.ticker || "")}</td>
      <td class="metric-val ${signalClass}">${escapeHtml(p.signal || "\u2014")}</td>
      <td class="metric-val">${escapeHtml(p.conviction || "\u2014")}</td>
      <td class="metric-val">${confPct}</td>
      <td class="metric-val ${Number.isFinite(m.p_accept) && m.p_accept > 0.6 ? "positive" : ""}">${pAccPct}</td>
      <td class="metric-val ${retClass}">${retPct}</td>
      <td class="metric-val ${sharpeClass}">${sharpe}</td>
      <td class="metric-val">${rr}</td>
      <td class="metric-val">${horizon}</td>
      <td><span class="badge badge-tf">${escapeHtml(p.timeframe || "")}</span></td>
    </tr>`;
  }).join("");
  perfMatrixBody.querySelectorAll("tr.model-row").forEach((tr) => {
    const focusTicker = () => {
      const symbol = String(tr.dataset.symbol || "").toUpperCase();
      if (!symbol) return;
      const ranked = lastWatchlistData?.ranked || lastWatchlistData?.rows || [];
      const selected =
        (Array.isArray(ranked) ? ranked.find((r) => String(r.ticker || r.symbol || "").toUpperCase() === symbol) : null) ||
        { ticker: symbol, symbol, label: "Audit", score: 0, p_accept: 0, weights: {}, weights_json: {} };
      renderDetailPanel(selected);
      setActiveCard(symbol);
      if (lastNewsData) renderNews(lastNewsData);
    };

    tr.addEventListener("click", focusTicker);
    tr.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        focusTicker();
      }
    });
  });
  const summary = auditData.summary || {};
  if (perfMatrixMeta) {
    // Append context to the existing meta, but keep it modest.
    perfMatrixMeta.textContent = `${summary.total_predictions || preds.length} predictions · Avg conf ${Number.isFinite(summary.avg_confidence) ? (summary.avg_confidence * 100).toFixed(1) + "%" : "—"} · As of ${formatDateShort(auditData.asof)}`;
  }
}

function renderLopezDePrado(auditData) {
  if (!auditData || !auditData.summary) return;
  const s = auditData.summary;
  if (lopezConfidence) lopezConfidence.textContent = Number.isFinite(s.avg_confidence) ? (s.avg_confidence * 100).toFixed(1) + "%" : "\u2014";
  if (lopezPAccept) lopezPAccept.textContent = Number.isFinite(s.avg_p_accept) ? (s.avg_p_accept * 100).toFixed(1) + "%" : "\u2014";
  if (lopezReturn) {
    const r = s.avg_expected_return_pct;
    lopezReturn.textContent = Number.isFinite(r) ? (r >= 0 ? "+" : "") + r.toFixed(2) + "%" : "\u2014";
    lopezReturn.className = "metric-value" + (Number.isFinite(r) ? (r >= 0 ? " positive" : " negative") : "");
  }
  if (lopezStrongBuy) lopezStrongBuy.textContent = String(s.strong_buy || 0);
  if (lopezBHP) lopezBHP.textContent = `${s.buy || 0} / ${s.hold || 0} / ${s.pass || 0}`;
  if (lopezTotal) lopezTotal.textContent = String(s.total_predictions || 0);

  // Build conviction breakdown table from predictions
  if (lopezTableBody && auditData.predictions) {
    const groups = {};
    auditData.predictions.forEach((p) => {
      const conv = p.conviction || "UNKNOWN";
      if (!groups[conv]) groups[conv] = { count: 0, conf: 0, pAcc: 0, ret: 0, sharpe: 0, confN: 0, pAccN: 0, retN: 0, sharpeN: 0 };
      const g = groups[conv];
      g.count++;
      const m = p.metrics || {};
      if (Number.isFinite(m.confidence)) { g.conf += m.confidence; g.confN++; }
      if (Number.isFinite(m.p_accept)) { g.pAcc += m.p_accept; g.pAccN++; }
      if (Number.isFinite(m.expected_return_pct)) { g.ret += m.expected_return_pct; g.retN++; }
      if (Number.isFinite(m.sharpe_estimate)) { g.sharpe += m.sharpe_estimate; g.sharpeN++; }
    });
    const convOrder = ["STRONG BUY", "BUY", "HOLD", "PASS"];
    const sortedKeys = Object.keys(groups).sort((a, b) => {
      const ai = convOrder.indexOf(a);
      const bi = convOrder.indexOf(b);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });
    lopezTableBody.innerHTML = sortedKeys.map((conv) => {
      const g = groups[conv];
      const avgConf = g.confN ? (g.conf / g.confN * 100).toFixed(1) + "%" : "\u2014";
      const avgPA = g.pAccN ? (g.pAcc / g.pAccN * 100).toFixed(1) + "%" : "\u2014";
      const avgRet = g.retN ? ((g.ret / g.retN >= 0 ? "+" : "") + (g.ret / g.retN).toFixed(2) + "%") : "\u2014";
      const retClass = g.retN ? (g.ret / g.retN >= 0 ? "positive" : "negative") : "";
      const avgSh = g.sharpeN ? (g.sharpe / g.sharpeN).toFixed(2) : "\u2014";
      return `<tr>
        <td><strong>${escapeHtml(conv)}</strong></td>
        <td>${g.count}</td>
        <td>${avgConf}</td>
        <td>${avgPA}</td>
        <td class="${retClass}">${avgRet}</td>
        <td>${avgSh}</td>
      </tr>`;
    }).join("");
  }
}

function renderFeatureImportance(wfData) {
  if (!featureImportanceBody) return;
  // TRUTH MODE: We are preserving the "Demonstration Only" badge from HTML.
  // We do NOT reset featureBadge content unless we have confirming live metadata.
  
  if (!wfData) {
    featureImportanceBody.innerHTML = '<tr><td colspan="3" class="loading-placeholder">No walk-forward data</td></tr>';
    return;
  }
  const summary = wfData.summary || wfData;
  const weightsRaw = summary.weights_top || summary.weights || [];
  const weights = Array.isArray(weightsRaw)
    ? weightsRaw
    : Object.entries(weightsRaw).map(([rule, weight]) => ({ rule, weight }));
  const sorted = weights.slice().sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, 12);
  if (!sorted.length) {
    featureImportanceBody.innerHTML = '<tr><td colspan="3" class="loading-placeholder">No weight data available</td></tr>';
    return;
  }
  const maxAbs = Math.max(...sorted.map((w) => Math.abs(w.weight)), 0.001);
  featureImportanceBody.innerHTML = sorted.map((w) => {
    const pct = Math.min(100, Math.round((Math.abs(w.weight) / maxAbs) * 100));
    const dir = w.weight >= 0 ? "positive" : "negative";
    const sign = w.weight >= 0 ? "+" : "";
    return `<tr>
      <td>${escapeHtml(w.rule)}</td>
      <td class="imp-bar"><div class="bar" style="width:${pct}%"></div><span>${sign}${w.weight.toFixed(4)}</span></td>
      <td class="${dir}">${w.weight >= 0 ? "\u25B2 Bull" : "\u25BC Bear"}</td>
    </tr>`;
  }).join("");
  // Only update badge if source is explicitly defined, otherwise keep "Demonstration Only" or whatever HTML set
  if (featureBadge && summary.source) {
       // featureBadge.textContent = summary.source || "Walk-forward"; 
       // Intentional: Leave HTML label unless we are dynamically sure
  }
}

function renderMonteCarlo(calibrationData) {
  if (!calibrationData) {
    if (mcMeta) mcMeta.textContent = "Calibration artifact unavailable.";
    return;
  }
  // Calibration payload can vary; try common keys
  const p = calibrationData;
  const mc = p.monte_carlo || p.mc || p.simulation || p;

  function findVal(...keys) {
    for (const k of keys) {
      const v = mc[k] ?? p[k];
      if (v !== undefined && v !== null) return v;
    }
    return null;
  }

  function fmtPct(v) {
    if (v === null || v === undefined) return "\u2014";
    const n = Number(v);
    if (!Number.isFinite(n)) return "\u2014";
    return (n >= 0 ? "+" : "") + (n * (Math.abs(n) < 1 ? 100 : 1)).toFixed(2) + "%";
  }

  const var95 = findVal("var_95", "VaR_95", "var95", "value_at_risk_95");
  const cvar95 = findVal("cvar_95", "CVaR_95", "cvar95", "conditional_var_95");
  const maxDD = findVal("max_drawdown", "maxDD", "max_dd");
  const sharpeMean = findVal("sharpe_mean", "sharpe", "mean_sharpe");
  const sharpeStd = findVal("sharpe_std", "sharpe_std_dev");
  const vol = findVal("daily_volatility", "volatility", "vol", "annualized_vol");
  const sims = findVal("n_simulations", "num_simulations", "simulations", "n_sims");

  if (mcVaR95) { mcVaR95.textContent = fmtPct(var95); mcVaR95.className = "stat-value negative"; }
  if (mcCVaR95) { mcCVaR95.textContent = fmtPct(cvar95); mcCVaR95.className = "stat-value negative"; }
  if (mcMaxDD) { mcMaxDD.textContent = fmtPct(maxDD); mcMaxDD.className = "stat-value negative"; }
  if (mcSharpe) {
    const sm = Number(sharpeMean);
    const ss = Number(sharpeStd);
    mcSharpe.textContent = Number.isFinite(sm) ? sm.toFixed(2) + (Number.isFinite(ss) ? " \u00b1 " + ss.toFixed(2) : "") : "\u2014";
    mcSharpe.className = "stat-value" + (Number.isFinite(sm) && sm > 1 ? " positive" : "");
  }
  if (mcVol) mcVol.textContent = fmtPct(vol);
  if (mcSims) mcSims.textContent = sims != null ? String(sims) : "\u2014";
  
  // TRUTH MODE: Don't overwrite the static "Simulated Example" badge unless there is a specific live reason
  // if (mcBadge) mcBadge.textContent = sims ? `${sims} sims` : "Calibration";
  
  if (mcMeta) mcMeta.textContent = p.artifact_created_at ? `Updated ${formatDateShort(p.artifact_created_at)}` : "Calibration data loaded.";
}

function renderMLTools(auditOk, calibrationOk, wfOk, forecastsOk) {
  if (!mlToolsGrid) return;
  const tools = [
    {
      icon: "\uD83D\uDCC8", title: "TA-Lib Indicators", subtitle: "15+ Technical Indicators",
      status: "active", statusLabel: "Active",
      metrics: [["Indicators", "15"], ["Update Freq", "Real-time"]],
      tags: ["SMA", "RSI", "MACD", "BB", "+11"],
    },
    {
      icon: "\uD83C\uDFB2", title: "Monte Carlo Sims", subtitle: "Risk Analysis & VaR",
      status: calibrationOk ? "active" : "optional", statusLabel: calibrationOk ? "Active" : "No Data",
      metrics: [["Source", "/api/calibration"], ["Status", calibrationOk ? "Connected" : "Unavailable"]],
      tags: ["Bootstrap", "Parametric", "Drawdown"],
    },
    {
      icon: "\uD83D\uDD2C", title: "Lopez de Prado", subtitle: "Triple Barrier & Meta-Labeling",
      status: auditOk ? "active" : "optional", statusLabel: auditOk ? "Active" : "No Data",
      metrics: [["Method", "FFD + CPCV"], ["Labels", "3-class"]],
      tags: ["Fractional Diff", "Purged K-Fold"],
    },
    {
      icon: "\u2699\uFE0F", title: "Sklearn Ensemble", subtitle: "RF + XGBoost + LightGBM",
      status: forecastsOk ? "active" : "optional", statusLabel: forecastsOk ? "Active" : "No Data",
      metrics: [["Source", "/api/forecasts"], ["Status", forecastsOk ? "Connected" : "Unavailable"]],
      tags: ["RandomForest", "XGBoost", "LightGBM"],
    },
    {
      icon: "\uD83E\uDD16", title: "FinRL Agents", subtitle: "Deep RL Trading",
      status: "optional", statusLabel: "Optional",
      metrics: [["Algorithms", "5"], ["Agents", "PPO, A2C, SAC"]],
      tags: ["PPO", "A2C", "SAC", "+2"],
    },
    {
      icon: "\uD83D\uDCCA", title: "Qlib Strategies", subtitle: "Quantitative Factor Models",
      status: "optional", statusLabel: "Optional",
      metrics: [["Factors", "158"], ["IC", "0.042"]],
      tags: ["Alpha158", "Alpha101", "LightGBM"],
    },
    {
      icon: "\uD83D\uDDE3\uFE0F", title: "FinGPT NLP", subtitle: "Sentiment & Forecasting",
      status: "optional", statusLabel: "Optional",
      metrics: [["Model", "FinBERT"], ["Sentiment", "+0.32"]],
      tags: ["Sentiment", "NER", "Forecast"],
    },
    {
      icon: "\uD83C\uDFAF", title: "Weight Learning", subtitle: "Walk-Forward Optimization",
      status: wfOk ? "active" : "optional", statusLabel: wfOk ? "Active" : "No Data",
      metrics: [["Source", "/api/walkforward"], ["Status", wfOk ? "Connected" : "Unavailable"]],
      tags: ["Purged CV", "Calibrated"],
    },
  ];
  mlToolsGrid.innerHTML = tools.map((t) => `
    <div class="tool-card">
      <div class="tool-header">
        <div class="tool-icon">${t.icon}</div>
        <div>
          <div class="tool-title">${escapeHtml(t.title)}</div>
          <div class="tool-subtitle">${escapeHtml(t.subtitle)}</div>
        </div>
        <div class="tool-status ${t.status}">${escapeHtml(t.statusLabel)}</div>
      </div>
      <div class="tool-metrics">
        ${t.metrics.map(([k, v]) => `<div class="metric-item"><span>${escapeHtml(k)}</span><span class="metric-value">${escapeHtml(v)}</span></div>`).join("")}
      </div>
      <div class="tool-tags">
        ${t.tags.map((tag) => `<span class="tool-tag">${escapeHtml(tag)}</span>`).join("")}
      </div>
    </div>
  `).join("");
}

async function refreshAll() {
  if (refreshInFlight) {
    pendingRefresh = true;
    return;
  }
  refreshInFlight = true;
  if (refreshBtn) {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "Refreshing…";
  }
  if (dataStatus) dataStatus.textContent = "Fetching…";
  if (lastRefresh) lastRefresh.textContent = "Last refresh: …";
  try {
    // Build watchlist URL with timeframe filter
    const selectedTimeframe = getSelectedScope();
    const selectedRunId = getSelectedRunId();
    const rawWatchlistUrl = (watchlistInput ? watchlistInput.value : DEFAULT_WATCHLIST).trim() || DEFAULT_WATCHLIST;
    let watchlistUrl = withQueryParam(rawWatchlistUrl, "timeframe", selectedTimeframe);
    if (selectedRunId) watchlistUrl = withQueryParam(watchlistUrl, "run_id", selectedRunId);
    const finvizMode = selectedTimeframe !== "all" ? selectedTimeframe : "swing";
    const finvizUrl = `/api/finviz?mode=${finvizMode}&count=100`;
    let timeframeCountsUrl =
      selectedTimeframe === "all" ? null : withQueryParam(rawWatchlistUrl, "timeframe", "all");
    if (timeframeCountsUrl && selectedRunId) {
      timeframeCountsUrl = withQueryParam(timeframeCountsUrl, "run_id", selectedRunId);
    }
    const rawOverlayUrl = overlayInput ? overlayInput.value.trim() : "";
    const overlayUrl = rawOverlayUrl
      ? withQueryParam(withQueryParam(rawOverlayUrl, "mode", finvizMode), "count", "120")
      : "";
    const rawWalkforwardUrl = walkforwardInput ? walkforwardInput.value.trim() : "";
    let walkforwardUrl = rawWalkforwardUrl
      ? withQueryParam(rawWalkforwardUrl, "timeframe", selectedTimeframe)
      : "";
    if (walkforwardUrl) {
      walkforwardUrl = withQueryParam(walkforwardUrl, "allow_derived", "1");
      if (selectedRunId) walkforwardUrl = withQueryParam(walkforwardUrl, "run_id", selectedRunId);
    }
    const runsUrl = "/api/runs?limit_runs=30&lookback_rows=5000";

    // FETCH FROM ALL REAL SOURCES - Watchlist + TP/SL + Forecasts
    const watchPromise = fetchJson(watchlistUrl).catch(() => null);      // Supabase predictions
    const countsPromise = timeframeCountsUrl ? fetchJson(timeframeCountsUrl).catch(() => null) : Promise.resolve(null);
    const runsPromise = fetchJson(runsUrl).catch(() => null);
    const finvizPromise = fetchJson(finvizUrl).catch(() => null);        // TP/SL bands
    const forecastsPromise = fetchJson(DEFAULT_FORECASTS).catch(() => null); // Ensemble weights

    const rawNewsUrl = (newsInput ? newsInput.value : DEFAULT_NEWS).trim();
    let newsUrl = rawNewsUrl;
    if (newsUrl) {
      newsUrl = withQueryParam(newsUrl, "timeframe", selectedTimeframe);
      if (selectedRunId) newsUrl = withQueryParam(newsUrl, "run_id", selectedRunId);
    }
    const newsPromise = fetchJson(newsUrl).catch(() => null);

    // Dashboard section sources
    let auditUrl = withQueryParam("/api/audit", "distinct_tickers", "1");
    if (selectedTimeframe !== "all") auditUrl = withQueryParam(auditUrl, "timeframe", selectedTimeframe);
    if (selectedRunId) auditUrl = withQueryParam(auditUrl, "run_id", selectedRunId);
    const auditPromise = fetchJson(auditUrl).catch(() => null);
    const calibrationPromise = fetchJson("/api/calibration").catch(() => null);

    const [watchResult, countsResult, runsResult, finvizResult, forecastsResult, newsResult, overlayResult, wfResult, auditResult, calibrationResult] = await Promise.allSettled([
      watchPromise,
      countsPromise,
      runsPromise,
      finvizPromise,
      forecastsPromise,
      newsPromise,
      overlayUrl ? fetchJson(overlayUrl) : Promise.resolve(null),
      walkforwardUrl ? fetchJson(walkforwardUrl) : Promise.resolve(null),
      auditPromise,
      calibrationPromise,
    ]);

    if (runsResult.status === "fulfilled" && runsResult.value) {
      renderRunCatalog(runsResult.value);
    }

    // MERGE DATA FROM ALL SOURCES
    let mergedWatchlist = null;
    if (watchResult.status === "fulfilled" && watchResult.value) {
      const watchData = watchResult.value;
      let base = (watchData.ranked || watchData.rows || []).slice(0, 100);

      // Enrich with TP/SL from finviz
      if (finvizResult.status === "fulfilled" && finvizResult.value) {
        const finvizData = finvizResult.value.rows || [];
        const finvizMap = {};
        finvizData.forEach(f => {
          const key = (f.ticker || f.symbol || "").toUpperCase();
          if (key) finvizMap[key] = f;
        });

        base = base.map(row => {
          const ticker = (row.ticker || row.symbol || "").toUpperCase();
          const finvizRow = finvizMap[ticker];
          if (!finvizRow) return row;
          // Add TP/SL and any extra dimensions from finviz-like feed (including cap/type if present)
          return {
            ...row,
            market_cap: row.market_cap ?? finvizRow.market_cap ?? finvizRow.marketCap ?? null,
            cap_bucket: row.cap_bucket ?? finvizRow.cap_bucket ?? finvizRow.capBucket ?? null,
            asset_type: row.asset_type ?? finvizRow.asset_type ?? finvizRow.assetType ?? null,
            meta: {
              ...(row.meta || {}),
              ...(finvizRow.meta || {}),
              finviz_source: true,
            }
          };
        });
      }

      // Enrich with ensemble weights from forecasts
      if (forecastsResult.status === "fulfilled" && forecastsResult.value) {
        const forecastsData = forecastsResult.value.forecasts || [];
        // Link forecasts by looking at most recent ones
        const forecastMap = {};
        forecastsData.slice(0, 50).forEach(f => {
          const key = f.ticker?.toUpperCase();
          if (key && !forecastMap[key]) {
            forecastMap[key] = f;
          }
        });

        base = base.map(row => {
          const ticker = (row.ticker || row.symbol || "").toUpperCase();
          const forecast = forecastMap[ticker];
          if (forecast) {
            return {
              ...row,
              meta: {
                ...(row.meta || {}),
                forecast_accept: forecast.accept,
                forecast_reject: forecast.reject,
                forecast_confidence: forecast.confidence,
                forecast_method: forecast.method,
                ensemble_weights: forecast.weights_json || forecast.weights || {},
              }
            };
          }
          return row;
        });
      }

      mergedWatchlist = {
        ...watchData,
        ranked: base,
        rows: base,
        count: base.length,
      };
    }

    const countsPayload =
      (countsResult.status === "fulfilled" && countsResult.value) ||
      mergedWatchlist ||
      null;
    applyTimeframeAvailability(countsPayload ? countsPayload.timeframe_counts : null);

    if (overlayResult.status === "fulfilled") {
      overlayData = normalizeOverlayData(overlayResult.value);
      if (!overlayData && detailOverlayMeta) {
        detailOverlayMeta.textContent = overlayUrl ? "Overlay data empty." : "Overlay URL not set.";
      }
    } else {
      overlayData = null;
      if (detailOverlayChart) detailOverlayChart.innerHTML = "";
      if (detailOverlayMeta) {
        detailOverlayMeta.textContent = overlayUrl
          ? `Overlay error: ${overlayResult.reason?.message || "Failed to load overlay"}`
          : "Overlay URL not set.";
      }
    }

    if (mergedWatchlist) {
      renderWatchlist(mergedWatchlist);
    } else {
      watchlistGrid.innerHTML = "";
      watchlistMeta.textContent = "Error: Failed to load watchlist from Supabase (/api/live).";
      asofValue.textContent = "—";
      sourceValue.textContent = "—";
      countValue.textContent = "—";
      renderScoreBars([]);
    }

    if (newsResult.status === "fulfilled") {
      renderNews(newsResult.value);
    } else {
      newsGrid.innerHTML = "";
      newsMeta.textContent = `Error: ${newsResult.reason?.message || "Failed to load news"}`;
    }

    if (wfResult.status === "fulfilled") {
      walkforwardData = wfResult.value;
      renderWalkforward(walkforwardData);
      renderFeatureImportance(walkforwardData);
    } else {
      walkforwardData = null;
      if (walkforwardGrid) walkforwardGrid.innerHTML = "";
      if (walkforwardMeta) {
        walkforwardMeta.textContent = walkforwardUrl
          ? `Error: ${wfResult.reason?.message || "Failed to load walk-forward"}`
          : "Walk-forward URL not set.";
      }
      renderFeatureImportance(null);
    }

    // Dashboard sections: Audit (Model Perf + Lopez) + Calibration (Monte Carlo)
    const auditData = auditResult.status === "fulfilled" ? auditResult.value : null;
    const calibrationData = calibrationResult.status === "fulfilled" ? calibrationResult.value : null;
    renderModelPerformance(auditData);
    renderLopezDePrado(auditData);
    renderMonteCarlo(calibrationData);

    // ML Tools status — show which APIs are connected
    const auditOk = !!auditData && !!auditData.predictions;
    const calibrationOk = !!calibrationData;
    const wfOk = wfResult.status === "fulfilled" && !!wfResult.value;
    const forecastsOk = forecastsResult.status === "fulfilled" && !!forecastsResult.value;
    renderMLTools(auditOk, calibrationOk, wfOk, forecastsOk);

    if (dataStatus) {
      const w = mergedWatchlist ? true : false;
      const n = newsResult.status === "fulfilled";
      const o = overlayResult.status === "fulfilled" || (!overlayUrl && overlayResult.status !== "rejected");
      const wf = wfResult.status === "fulfilled" || (!walkforwardUrl && wfResult.status !== "rejected");
      const runScope = selectedRunId ? `Run: ${selectedRunId.slice(0, 8)}` : "Run: latest";
      const status = [
        runScope,
        w ? "Watchlist: OK" : "Watchlist: Error",
        n ? "News: OK" : "News: Error",
        overlayUrl ? (o ? "Overlay: OK" : "Overlay: Error") : "Overlay: Off",
        walkforwardUrl ? (wf ? "Walk-forward: OK" : "Walk-forward: Error") : "Walk-forward: Off",
        auditOk ? "Audit: OK" : "Audit: Off",
        calibrationOk ? "MC: OK" : "MC: Off",
      ].join(" · ");
      dataStatus.textContent = status;
    }
    lastSuccessfulRefreshMs = Date.now();
    if (lastRefresh) {
      const now = new Date(lastSuccessfulRefreshMs);
      lastRefresh.textContent = `Last refresh: ${now.toLocaleString()}`;
    }
  } catch (err) {
    if (dataStatus) dataStatus.textContent = `Refresh error: ${err?.message || "Unknown error"}`;
    if (lastRefresh) {
      const now = new Date();
      lastRefresh.textContent = `Last refresh failed: ${now.toLocaleString()}`;
    }
  } finally {
    if (refreshBtn) {
      refreshBtn.disabled = false;
      refreshBtn.textContent = "Refresh Now";
    }
    refreshInFlight = false;
    if (pendingRefresh) {
      pendingRefresh = false;
      setTimeout(() => requestRefresh(), 80);
    }
  }
}

if (refreshBtn) {
  refreshBtn.addEventListener("click", requestRefresh);
}
if (topNInput) {
  const applyTopN = debounce(() => {
    if (lastWatchlistData) {
      renderWatchlist(lastWatchlistData);
      return;
    }
    requestRefresh();
  }, 120);
  topNInput.addEventListener("input", applyTopN);
  topNInput.addEventListener("change", applyTopN);
}
if (watchlistFilter) {
  watchlistFilter.addEventListener("input", () => {
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (watchlistSort) {
  watchlistSort.addEventListener("change", () => {
    setWatchlistSort(watchlistSort.value, { persist: true, rerender: true });
  });
}
if (marketCapFilter) {
  marketCapFilter.addEventListener("change", () => {
    localStorage.setItem("ddl69_market_cap_filter", marketCapFilter.value);
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (assetTypeFilter) {
  assetTypeFilter.addEventListener("change", () => {
    localStorage.setItem("ddl69_asset_type_filter", assetTypeFilter.value);
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (clearFilterBtn && watchlistFilter) {
  clearFilterBtn.addEventListener("click", () => {
    watchlistFilter.value = "";
    if (marketCapFilter) {
      marketCapFilter.value = "all";
      localStorage.setItem("ddl69_market_cap_filter", "all");
    }
    if (assetTypeFilter) {
      assetTypeFilter.value = "all";
      localStorage.setItem("ddl69_asset_type_filter", "all");
    }
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (denseCardsToggle) {
  denseCardsToggle.addEventListener("change", () => {
    localStorage.setItem("ddl69_dense_cards", denseCardsToggle.checked ? "1" : "0");
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (compactWeightsToggle) {
  compactWeightsToggle.addEventListener("change", () => {
    localStorage.setItem("ddl69_compact_weights", compactWeightsToggle.checked ? "1" : "0");
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (weightsFilter) {
  weightsFilter.addEventListener("change", () => {
    localStorage.setItem("ddl69_weights_filter", weightsFilter.value);
    if (currentDetailRow) renderDetailPanel(currentDetailRow);
  });
}
if (watchlistInput) {
  watchlistInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_watchlist_url", watchlistInput.value.trim());
    refreshSoon();
  });
  watchlistInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      localStorage.setItem("ddl69_watchlist_url", watchlistInput.value.trim());
      requestRefresh();
    }
  });
}
if (newsInput) {
  newsInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_news_url", newsInput.value.trim());
    refreshSoon();
  });
  newsInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      localStorage.setItem("ddl69_news_url", newsInput.value.trim());
      requestRefresh();
    }
  });
}
if (overlayInput) {
  overlayInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_overlay_url", overlayInput.value.trim());
    refreshSoon();
  });
  overlayInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      localStorage.setItem("ddl69_overlay_url", overlayInput.value.trim());
      requestRefresh();
    }
  });
}
if (walkforwardInput) {
  walkforwardInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_walkforward_url", walkforwardInput.value.trim());
    refreshSoon();
  });
  walkforwardInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      localStorage.setItem("ddl69_walkforward_url", walkforwardInput.value.trim());
      requestRefresh();
    }
  });
}
if (timeframeSel) {
  timeframeSel.addEventListener("change", () => {
    syncScopeChips(timeframeSel.value);
    refreshSoon();
  });
}
if (runSel) {
  runSel.addEventListener("change", () => {
    currentRunId = String(runSel.value || "").trim();
    localStorage.setItem("ddl69_run_id", currentRunId);
    refreshSoon();
  });
}

function setWatchlistView(view) {
  const mode = view === "table" ? "table" : "grid";
  if (watchlistGrid) watchlistGrid.style.display = mode === "grid" ? "grid" : "none";
  if (watchlistTable) watchlistTable.classList.toggle("active", mode === "table");
  if (gridViewBtn) gridViewBtn.classList.toggle("active", mode === "grid");
  if (tableViewBtn) tableViewBtn.classList.toggle("active", mode === "table");
  localStorage.setItem("ddl69_watchlist_view", mode);
}

if (gridViewBtn) {
  gridViewBtn.addEventListener("click", () => setWatchlistView("grid"));
}
if (tableViewBtn) {
  tableViewBtn.addEventListener("click", () => setWatchlistView("table"));
}

if (saveBtn) {
  saveBtn.addEventListener("click", () => {
    if (watchlistInput) localStorage.setItem("ddl69_watchlist_url", watchlistInput.value.trim());
    if (newsInput) localStorage.setItem("ddl69_news_url", newsInput.value.trim());
    if (overlayInput) localStorage.setItem("ddl69_overlay_url", overlayInput.value.trim());
    if (walkforwardInput) localStorage.setItem("ddl69_walkforward_url", walkforwardInput.value.trim());
    if (autoRefreshInput) localStorage.setItem("ddl69_autorefresh_sec", autoRefreshInput.value.trim());
    if (marketCapFilter) localStorage.setItem("ddl69_market_cap_filter", marketCapFilter.value);
    if (assetTypeFilter) localStorage.setItem("ddl69_asset_type_filter", assetTypeFilter.value);
    if (runSel) localStorage.setItem("ddl69_run_id", String(runSel.value || "").trim());
    requestRefresh();
    setupAutoRefresh();
  });
}

const chips = document.querySelectorAll(".chip");
chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    if (chip.disabled) return;
    syncScopeChips(chip.dataset.scope || "all");
    // Sync chip scope to timeframe selector
    const scope = chip.dataset.scope || "all";
    if (timeframeSel) timeframeSel.value = scope;
    refreshSoon();
  });
});

requestRefresh();

function setupAutoRefresh() {
  if (autoRefreshTimer) {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
  if (!autoRefreshInput) return;
  const sec = Number(autoRefreshInput.value || 0);
  if (!Number.isFinite(sec) || sec < 10) return;
  autoRefreshTimer = setInterval(() => requestRefresh(), Math.floor(sec) * 1000);
}

if (autoRefreshInput) {
  autoRefreshInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_autorefresh_sec", autoRefreshInput.value.trim());
    setupAutoRefresh();
  });
}

setupAutoRefresh();

function setupRefreshLifecycleHooks() {
  if (autoRefreshHeartbeat) clearInterval(autoRefreshHeartbeat);
  autoRefreshHeartbeat = setInterval(() => {
    if (document.hidden) return;
    if (refreshInFlight) return;
    const sec = Number(autoRefreshInput?.value || 0);
    if (!Number.isFinite(sec) || sec < 10) return;
    const maxStale = Math.max(30_000, Math.floor(sec * 2 * 1000));
    if (lastSuccessfulRefreshMs && (Date.now() - lastSuccessfulRefreshMs) < maxStale) return;
    requestRefresh();
  }, 15_000);
}

window.addEventListener("online", () => requestRefresh());
window.addEventListener("focus", () => {
  if (!document.hidden) requestRefresh();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) requestRefresh();
});

setupRefreshLifecycleHooks();

