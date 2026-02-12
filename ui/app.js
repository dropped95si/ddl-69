// PRIMARY SOURCES - Real trading system (Supabase + ML Pipeline)
const DEFAULT_WATCHLIST = "/api/live";           // Supabase ensemble predictions (strict mode)
const DEFAULT_FORECASTS = "/api/forecasts";      // Ensemble forecast time series
const DEFAULT_FINVIZ = "/api/finviz?mode=swing&count=100";  // TP/SL heuristics
const DEFAULT_OVERLAYS = "/api/overlays";        // Technical overlay charts
const DEFAULT_WALKFORWARD = "/api/walkforward";  // Walk-forward backtesting
const DEFAULT_NEWS = "/api/news";                // News + sentiment from Supabase/Polygon
const DEFAULT_OVERLAY = "/api/overlays";
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

const storedSort = localStorage.getItem("ddl69_watchlist_sort") || "score";
const storedAutoRefresh = localStorage.getItem("ddl69_autorefresh_sec") || "300";
const storedDense = localStorage.getItem("ddl69_dense_cards") || "0";
const storedView = localStorage.getItem("ddl69_watchlist_view") || "grid";
const storedCompactWeights = localStorage.getItem("ddl69_compact_weights") || "0";
const storedWeightsFilter = localStorage.getItem("ddl69_weights_filter") || "top";
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
if (watchlistSort) watchlistSort.value = storedSort;
if (autoRefreshInput) autoRefreshInput.value = storedAutoRefresh;
if (denseCardsToggle) denseCardsToggle.checked = storedDense === "1";
if (compactWeightsToggle) compactWeightsToggle.checked = storedCompactWeights === "1";
if (weightsFilter) weightsFilter.value = storedWeightsFilter;
setWatchlistView(storedView);

let overlayData = null;
let lastWatchlistData = null;
let walkforwardData = null;
let currentDetailRow = null;
let autoRefreshTimer = null;
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

const refreshSoon = debounce(() => {
  refreshAll();
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

function formatPct(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
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

function renderScoreBars(ranked) {
  if (!ranked || ranked.length === 0) {
    scoreBars.innerHTML = `<div class="small-note">No ranked scores available yet.</div>`;
    return;
  }
  const scores = ranked
    .map((r) => Number(r.score || r.p_accept || 0))
    .filter((v) => !Number.isNaN(v))
    .slice(0, Math.min(12, ranked.length));
  scoreBars.innerHTML = scores
    .map((s, i) => scoreBar(`Top ${i + 1}`, s))
    .join("");
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
  const sortMode = (watchlistSort?.value || "score").toLowerCase();

  const rankedList = data.ranked || data.rows || [];
  if (Array.isArray(rankedList)) {
    if (rankedList.length === 0) {
      countValue.textContent = 0;
      watchlistMeta.textContent = data.message || "Ranked list · no rows for this timeframe.";
      renderScoreBars([]);
      clearDetailPanel();
      renderWatchlistTable([]);
      return;
    }
    const base = rankedList;
    const filtered = filterTerm
      ? base.filter((row) => {
          const ticker = String(row.ticker || row.symbol || "").toUpperCase();
          const label = String(row.label || "").toUpperCase();
          const plan = String(row.plan_type || "").toUpperCase();
          return ticker.includes(filterTerm) || label.includes(filterTerm) || plan.includes(filterTerm);
        })
      : base;
    const sorted = [...filtered].sort((a, b) => {
      if (sortMode === "accept") {
        return Number(b.p_accept || 0) - Number(a.p_accept || 0);
      }
      if (sortMode === "ticker") {
        const at = String(a.ticker || a.symbol || "");
        const bt = String(b.ticker || b.symbol || "");
        return at.localeCompare(bt);
      }
      return Number(b.score || 0) - Number(a.score || 0);
    });
    const ranked = sorted.slice(0, topN);
    countValue.textContent = rankedList.length;
    watchlistMeta.textContent = `Ranked list · showing ${ranked.length}${filterTerm ? " (filtered)" : ""}`;
    renderScoreBars(base);
    if (ranked.length) {
      const selectedSymbol = String(currentDetailRow?.ticker || currentDetailRow?.symbol || "").toUpperCase();
      const selectedRow = ranked.find(
        (row) => String(row.ticker || row.symbol || "").toUpperCase() === selectedSymbol
      ) || ranked[0];
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
        <div class="watch-meta"><span>Timeframe</span><span class="badge badge-tf">${plan}</span></div>
        <div class="weight-list">${buildWeightsHtml(row.weights || row.weights_json || {})}</div>
      `;
      card.addEventListener("click", () => {
        openSymbolModal(row);
        setActiveCard(symbolRaw);
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
    const base = data.tickers;
    const filtered = filterTerm
      ? base.filter((t) => String(t).toUpperCase().includes(filterTerm))
      : base;
    const tickersSorted = sortMode === "ticker" ? [...filtered].sort() : filtered;
    const tickers = tickersSorted.slice(0, topN);
    countValue.textContent = data.count || data.tickers.length;
    watchlistMeta.textContent = `Universe list · showing ${tickers.length}${filterTerm ? " (filtered)" : ""}. No probabilities in this file.`;
    renderScoreBars([]);
    if (tickers.length) {
      renderDetailPanel({ ticker: tickers[0], label: "Universe", p_accept: 0, score: 0, weights: {} });
    } else {
      clearDetailPanel();
    }
    tickers.forEach((t) => {
      const ticker = escapeHtml(t);
      const row = { ticker: t, label: "Universe", p_accept: 0, score: 0, weights: {} };
      const card = document.createElement("div");
      card.className = "watch-card";
      card.dataset.symbol = t;
      card.innerHTML = `
        <h4>${ticker}</h4>
        <div class="watch-meta"><span>Universe</span><span>Member</span></div>
        <div class="small-note">No ranking data in this list. Use a ranked watchlist JSON.</div>
      `;
      card.addEventListener("click", () => {
        openSymbolModal(row);
        setActiveCard(t);
      });
      watchlistGrid.appendChild(card);
    });
    if (tickers.length) setActiveCard(tickers[0]);
    renderWatchlistTable(tickers.map((t) => ({ ticker: t, label: "Universe", p_accept: 0, score: 0, weights: {} })));
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

  const renderDetail = (row) => {
    const score = Number(row.score || 0);
    const accept = Number(row.p_accept || 0);
    const hit = getHitProbability(row);
    const scenarios = getScenarios(row);
    const target = getTargetPrice(row) || scenarios.up.tps[0] || null;
    const weights = row.weights || row.weights_json || {};
    const meta = getMeta(row);
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
      const symbol = escapeHtml(symbolRaw || "—");
      const plan = escapeHtml(row.plan_type || "—");
      const score = `${(Number(row.score || 0) * 100).toFixed(1)}%`;
      const accept = `${(Number(row.p_accept || 0) * 100).toFixed(1)}%`;
      const hit = `${(getHitProbability(row) * 100).toFixed(1)}%`;
      const scenarios = getScenarios(row);
      const target = getTargetPrice(row) || scenarios.up.tps[0] || null;
      const targetTxt = target ? `$${Number(target).toFixed(2)}` : "—";
      return `
        <tr class="main-row" data-symbol="${escapeHtml(symbolRaw)}">
          <td><span class="caret">?</span>${symbol}</td>
          <td>${score}</td>
          <td>${accept}</td>
          <td>${hit}</td>
          <td>${targetTxt}</td>
          <td>${plan}</td>
        </tr>
        <tr class="detail-row hidden" data-symbol="${escapeHtml(symbolRaw)}">
          <td colspan="6">${renderDetail(row)}</td>
        </tr>
      `;
    })
    .join("");

  watchlistTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Score</th>
          <th>P(accept)</th>
          <th>P(hit target)</th>
          <th>Target</th>
          <th>Plan</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;

  watchlistTable.querySelectorAll("tbody tr.main-row").forEach((tr) => {
    tr.addEventListener("click", () => {
      const symbol = tr.dataset.symbol || "";
      const row = rows.find((r) => String(r.ticker || r.symbol || "") === symbol);
      const detailRow = tr.nextElementSibling;
      const caret = tr.querySelector(".caret");
      if (detailRow && detailRow.classList.contains("detail-row")) {
        const isHidden = detailRow.classList.toggle("hidden");
        caret.textContent = isHidden ? "?" : "?";
        if (!detailRow.dataset.rendered && !isHidden) {
          const mini = detailRow.querySelector(".mini-chart");
          if (mini) renderMiniOverlayChart(mini, symbol, row);
          detailRow.dataset.rendered = "1";
        }
      }
      if (row) {
        renderDetailPanel(row);
        setActiveCard(symbol);
      }
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
  newsGrid.innerHTML = "";
  if (!data) {
    newsMeta.textContent = "No news data.";
    return;
  }
  const items = data.results || data.data || data.items || data.news || data;
  if (!Array.isArray(items)) {
    newsMeta.textContent = "Unsupported news format.";
    return;
  }

  const top = items.slice(0, 12);
  newsMeta.textContent = `Showing ${top.length} of ${items.length}`;
  top.forEach((item) => {
    const title = escapeHtml(item.title || item.headline || "Untitled");
    const ts = item.published_utc || item.timestamp || item.time || item.date;
    const tickers = item.tickers || item.symbols || [];
    const url = safeUrl(item.article_url || item.url || "#");
    const sentiment = item.sentiment || item.sentiment_score || item.score;
    const tickersDisplay = Array.isArray(tickers)
      ? escapeHtml(tickers.slice(0, 3).join(", "))
      : escapeHtml(tickers);
    const domainLabel = (() => {
      if (!url || url === "#") return "";
      try {
        const u = new URL(url);
        return u.hostname.replace("www.", "");
      } catch (err) {
        return "";
      }
    })();

    const card = document.createElement("div");
    card.className = "news-card";
    card.innerHTML = `
      <h5>${title}</h5>
      <div class="news-meta">
        <span>${formatDate(ts)}</span>
        <span>${tickersDisplay}</span>
      </div>
      <div class="news-meta">
        <span>Sentiment</span>
        <span>${sentiment !== undefined ? Number(sentiment).toFixed(2) : "—"}</span>
      </div>
      <div class="small-note"><a href="${url}" target="_blank" rel="noreferrer">${domainLabel || "Open source"}</a></div>
    `;
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
      <div class="wf-value">${formatPct(stats.net_weight || 0)}</div>
      <div class="wf-small">Avg win ${formatPct(stats.avg_win_rate || 0)} · Avg ret ${formatPct(stats.avg_return || 0)}</div>
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
    chip.disabled = false;
    chip.classList.remove("chip-disabled");
    chip.classList.toggle("chip-empty", unavailable);
    chip.setAttribute("aria-disabled", "false");
    chip.title = unavailable ? `No ${scope} rows in current run` : `${counts[scope]} ${scope} rows`;
  });

  if (timeframeSel) {
    Array.from(timeframeSel.options).forEach((opt) => {
      opt.disabled = false;
    });
  }

  return counts;
}

async function refreshAll() {
  if (refreshBtn) {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "Refreshing…";
  }
  if (dataStatus) dataStatus.textContent = "Fetching…";
  if (lastRefresh) lastRefresh.textContent = "Last refresh: …";

  // Build watchlist URL with timeframe filter
  const selectedTimeframe = getSelectedScope();
  const rawWatchlistUrl = (watchlistInput ? watchlistInput.value : DEFAULT_WATCHLIST).trim() || DEFAULT_WATCHLIST;
  const watchlistUrl = withQueryParam(rawWatchlistUrl, "timeframe", selectedTimeframe);
  const finvizMode = selectedTimeframe !== "all" ? selectedTimeframe : "swing";
  const finvizUrl = `/api/finviz?mode=${finvizMode}&count=100`;
  const timeframeCountsUrl =
    selectedTimeframe === "all" ? null : withQueryParam(rawWatchlistUrl, "timeframe", "all");
  const rawOverlayUrl = overlayInput ? overlayInput.value.trim() : "";
  const overlayUrl = rawOverlayUrl
    ? withQueryParam(withQueryParam(rawOverlayUrl, "mode", finvizMode), "count", "120")
    : "";
  const rawWalkforwardUrl = walkforwardInput ? walkforwardInput.value.trim() : "";
  const walkforwardUrl = rawWalkforwardUrl
    ? withQueryParam(rawWalkforwardUrl, "timeframe", selectedTimeframe)
    : "";

  // FETCH FROM ALL REAL SOURCES - Watchlist + TP/SL + Forecasts
  const watchPromise = fetchJson(watchlistUrl).catch(() => null);      // Supabase predictions
  const countsPromise = timeframeCountsUrl ? fetchJson(timeframeCountsUrl).catch(() => null) : Promise.resolve(null);
  const finvizPromise = fetchJson(finvizUrl).catch(() => null);        // TP/SL bands
  const forecastsPromise = fetchJson(DEFAULT_FORECASTS).catch(() => null); // Ensemble weights

  const newsUrl = (newsInput ? newsInput.value : DEFAULT_NEWS).trim();
  const newsPromise = fetchJson(newsUrl).catch(() => null);

  const [watchResult, countsResult, finvizResult, forecastsResult, newsResult, overlayResult, wfResult] = await Promise.allSettled([
    watchPromise,
    countsPromise,
    finvizPromise,
    forecastsPromise,
    newsPromise,
    overlayUrl ? fetchJson(overlayUrl) : Promise.resolve(null),
    walkforwardUrl ? fetchJson(walkforwardUrl) : Promise.resolve(null),
  ]);

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
        finvizMap[f.ticker.toUpperCase()] = f;
      });
      
      base = base.map(row => {
        const ticker = (row.ticker || row.symbol || "").toUpperCase();
        const finvizRow = finvizMap[ticker];
        if (finvizRow && finvizRow.meta) {
          // Add TP/SL and probability info
          return {
            ...row,
            meta: {
              ...(row.meta || {}),
              ...finvizRow.meta, // TP1-3, SL1-3, p_up, p_down, reason
              finviz_source: true,
            }
          };
        }
        return row;
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
  } else {
    walkforwardData = null;
    if (walkforwardGrid) walkforwardGrid.innerHTML = "";
    if (walkforwardMeta) {
      walkforwardMeta.textContent = walkforwardUrl
        ? `Error: ${wfResult.reason?.message || "Failed to load walk-forward"}`
        : "Walk-forward URL not set.";
    }
  }

  if (dataStatus) {
    const w = mergedWatchlist ? true : false;
    const n = newsResult.status === "fulfilled";
    const o = overlayResult.status === "fulfilled" || (!overlayUrl && overlayResult.status !== "rejected");
    const wf = wfResult.status === "fulfilled" || (!walkforwardUrl && wfResult.status !== "rejected");
    const status = [
      w ? "Watchlist: OK" : "Watchlist: Error",
      n ? "News: OK" : "News: Error",
      overlayUrl ? (o ? "Overlay: OK" : "Overlay: Error") : "Overlay: Off",
      walkforwardUrl ? (wf ? "Walk-forward: OK" : "Walk-forward: Error") : "Walk-forward: Off",
    ].join(" · ");
    dataStatus.textContent = status;
  }
  if (lastRefresh) {
    const now = new Date();
    lastRefresh.textContent = `Last refresh: ${now.toLocaleString()}`;
  }

  if (refreshBtn) {
    refreshBtn.disabled = false;
    refreshBtn.textContent = "Refresh Now";
  }
}

if (refreshBtn) {
  refreshBtn.addEventListener("click", refreshAll);
}
if (topNInput) {
  topNInput.addEventListener("change", refreshAll);
}
if (watchlistFilter) {
  watchlistFilter.addEventListener("input", () => {
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (watchlistSort) {
  watchlistSort.addEventListener("change", () => {
    localStorage.setItem("ddl69_watchlist_sort", watchlistSort.value);
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (clearFilterBtn && watchlistFilter) {
  clearFilterBtn.addEventListener("click", () => {
    watchlistFilter.value = "";
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
}
if (newsInput) {
  newsInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_news_url", newsInput.value.trim());
    refreshSoon();
  });
}
if (overlayInput) {
  overlayInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_overlay_url", overlayInput.value.trim());
    refreshSoon();
  });
}
if (walkforwardInput) {
  walkforwardInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_walkforward_url", walkforwardInput.value.trim());
    refreshSoon();
  });
}
if (timeframeSel) {
  timeframeSel.addEventListener("change", () => {
    syncScopeChips(timeframeSel.value);
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
    refreshAll();
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

refreshAll();

function setupAutoRefresh() {
  if (autoRefreshTimer) {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
  if (!autoRefreshInput) return;
  const sec = Number(autoRefreshInput.value || 0);
  if (!Number.isFinite(sec) || sec < 10) return;
  autoRefreshTimer = setInterval(refreshAll, Math.floor(sec) * 1000);
}

if (autoRefreshInput) {
  autoRefreshInput.addEventListener("change", () => {
    localStorage.setItem("ddl69_autorefresh_sec", autoRefreshInput.value.trim());
    setupAutoRefresh();
  });
}

setupAutoRefresh();

