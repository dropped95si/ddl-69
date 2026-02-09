const DEFAULT_WATCHLIST = "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/watchlist/watchlist_2026-02-08.json";
const DEFAULT_NEWS = "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/news/polygon_news_2026-02-08.json";
const DEFAULT_OVERLAY = "";

const watchlistInput = document.getElementById("watchlistUrl");
const newsInput = document.getElementById("newsUrl");
const overlayInput = document.getElementById("overlayUrl");
const refreshBtn = document.getElementById("refreshBtn");
const saveBtn = document.getElementById("saveBtn");
const topNInput = document.getElementById("topN");

const asofValue = document.getElementById("asofValue");
const sourceValue = document.getElementById("sourceValue");
const countValue = document.getElementById("countValue");
const watchlistMeta = document.getElementById("watchlistMeta");
const watchlistGrid = document.getElementById("watchlistGrid");
const watchlistFilter = document.getElementById("watchlistFilter");
const clearFilterBtn = document.getElementById("clearFilter");
const scoreBars = document.getElementById("scoreBars");
const newsMeta = document.getElementById("newsMeta");
const newsGrid = document.getElementById("newsGrid");

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
const detailChart = document.getElementById("detailChart");
const detailOverlayChart = document.getElementById("detailOverlayChart");
const detailOverlayMeta = document.getElementById("detailOverlayMeta");
const detailOverlaySummary = document.getElementById("detailOverlaySummary");

const storedWatchlist = localStorage.getItem("ddl69_watchlist_url") || DEFAULT_WATCHLIST;
const storedNews = localStorage.getItem("ddl69_news_url") || DEFAULT_NEWS;
const storedOverlay = localStorage.getItem("ddl69_overlay_url") || DEFAULT_OVERLAY;
watchlistInput.value = storedWatchlist;
newsInput.value = storedNews;
if (overlayInput) overlayInput.value = storedOverlay;

let overlayData = null;
let lastWatchlistData = null;

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

function formatDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toISOString().replace("T", " ").replace("Z", " UTC");
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

function resolveLineStyle(style) {
  const lw = window.LightweightCharts;
  if (!lw || !lw.LineStyle) return 0;
  const s = String(style || "solid").toLowerCase();
  if (s === "dotted") return lw.LineStyle.Dotted;
  if (s === "dashed") return lw.LineStyle.Dashed;
  return lw.LineStyle.Solid;
}

let overlayChart = null;
let overlayResizeObserver = null;

function renderOverlayChart(symbol) {
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
    .slice(0, 5);
  scoreBars.innerHTML = scores
    .map((s, i) => scoreBar(`Top ${i + 1}`, s))
    .join("");
}

function buildWeightsHtml(weights) {
  if (!weights || Object.keys(weights).length === 0) {
    return `<div class="small-note">No weights available</div>`;
  }
  return Object.entries(weights)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 8)
    .map(
      ([k, v]) =>
        `<div class="weight-item"><span>${escapeHtml(k)}</span><span>${(Number(v) * 100).toFixed(1)}%</span></div>`
    )
    .join("");
}

function renderTradingView(symbol) {
  if (!tvChart) return;
  tvChart.innerHTML = `
    <iframe
      src="https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=D&theme=dark&style=1&locale=en&toolbarbg=%23070b15&hide_side_toolbar=1&withdateranges=1&allow_symbol_change=0"
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
  const symbol = row.ticker || row.symbol || "—";
  const score = Number(row.score || 0);
  const prob = Number(row.p_accept || 0);

  if (detailSymbol) detailSymbol.textContent = symbol;
  if (detailScore) detailScore.textContent = `${(score * 100).toFixed(1)}% score`;
  if (detailProb) detailProb.textContent = `${(prob * 100).toFixed(1)}% accept`;
  if (detailLabel) detailLabel.textContent = row.label || "—";
  if (detailWeights) detailWeights.innerHTML = buildWeightsHtml(row.weights || row.weights_json || {});
  if (detailChart) {
    detailChart.innerHTML = `
      <iframe
        src="https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=D&theme=dark&style=1&locale=en&toolbarbg=%23070b15&hide_side_toolbar=1&withdateranges=1&allow_symbol_change=0"
        width="100%"
        height="340"
        frameborder="0"
        allowtransparency="true"
        loading="lazy"
      ></iframe>
    `;
  }
  renderOverlayChart(symbol);
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
  renderTradingView(symbol);

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

function renderWatchlist(data) {
  const topN = Number(topNInput.value || 10);
  watchlistGrid.innerHTML = "";
  if (!data) {
    watchlistMeta.textContent = "No watchlist data.";
    return;
  }
  lastWatchlistData = data;

  const asof = data.asof || data.generated_at || "";
  asofValue.textContent = formatDate(asof);
  sourceValue.textContent = data.source || data.provider || "Supabase artifacts";

  const filterTerm = (watchlistFilter?.value || "").trim().toUpperCase();

  if (Array.isArray(data.ranked) && data.ranked.length) {
    const base = data.ranked;
    const filtered = filterTerm
      ? base.filter((row) => {
          const ticker = String(row.ticker || row.symbol || "").toUpperCase();
          const label = String(row.label || "").toUpperCase();
          const plan = String(row.plan_type || "").toUpperCase();
          return ticker.includes(filterTerm) || label.includes(filterTerm) || plan.includes(filterTerm);
        })
      : base;
    const ranked = filtered.slice(0, topN);
    countValue.textContent = data.ranked.length;
    watchlistMeta.textContent = `Ranked list · showing ${ranked.length}${filterTerm ? " (filtered)" : ""}`;
    renderScoreBars(base);
    if (ranked.length) {
      renderDetailPanel(ranked[0]);
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
      card.innerHTML = `
        <h4>${symbol}</h4>
        <div class="watch-meta"><span>${label}</span><span>Score ${(Number(row.score || 0) * 100).toFixed(1)}%</span></div>
        <div class="watch-meta"><span>P(accept)</span><span>${(Number(row.p_accept || 0) * 100).toFixed(1)}%</span></div>
        <div class="watch-meta"><span>Plan</span><span>${plan}</span></div>
        <div class="weight-list">${buildWeightsHtml(row.weights || row.weights_json || {})}</div>
      `;
      card.addEventListener("click", () => {
        openSymbolModal(row);
        setActiveCard(symbolRaw);
      });
      watchlistGrid.appendChild(card);
    });
    if (ranked.length) setActiveCard(ranked[0].ticker || ranked[0].symbol || "—");
    return;
  }

  if (Array.isArray(data.tickers)) {
    const base = data.tickers;
    const filtered = filterTerm
      ? base.filter((t) => String(t).toUpperCase().includes(filterTerm))
      : base;
    const tickers = filtered.slice(0, topN);
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
    return;
  }

  watchlistMeta.textContent = "Unsupported watchlist format.";
}

function clearDetailPanel() {
  if (detailSymbol) detailSymbol.textContent = "—";
  if (detailScore) detailScore.textContent = "—";
  if (detailProb) detailProb.textContent = "—";
  if (detailLabel) detailLabel.textContent = "—";
  if (detailWeights) detailWeights.innerHTML = "";
  if (detailChart) detailChart.innerHTML = "";
  if (detailOverlayChart) detailOverlayChart.innerHTML = "";
  if (detailOverlayMeta) detailOverlayMeta.textContent = "Overlay data pending.";
  if (detailOverlaySummary) detailOverlaySummary.innerHTML = "";
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
      <div class="small-note"><a href="${url}" target="_blank" rel="noreferrer">Open source</a></div>
    `;
    newsGrid.appendChild(card);
  });
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

async function refreshAll() {
  const overlayUrl = overlayInput ? overlayInput.value.trim() : "";
  const [watchResult, newsResult, overlayResult] = await Promise.allSettled([
    fetchJson(watchlistInput.value.trim()),
    fetchJson(newsInput.value.trim()),
    overlayUrl ? fetchJson(overlayUrl) : Promise.resolve(null),
  ]);

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

  if (watchResult.status === "fulfilled") {
    renderWatchlist(watchResult.value);
  } else {
    watchlistGrid.innerHTML = "";
    watchlistMeta.textContent = `Error: ${watchResult.reason?.message || "Failed to load watchlist"}`;
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
}

refreshBtn.addEventListener("click", refreshAll);
topNInput.addEventListener("change", refreshAll);
if (watchlistFilter) {
  watchlistFilter.addEventListener("input", () => {
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}
if (clearFilterBtn && watchlistFilter) {
  clearFilterBtn.addEventListener("click", () => {
    watchlistFilter.value = "";
    if (lastWatchlistData) renderWatchlist(lastWatchlistData);
  });
}

saveBtn.addEventListener("click", () => {
  localStorage.setItem("ddl69_watchlist_url", watchlistInput.value.trim());
  localStorage.setItem("ddl69_news_url", newsInput.value.trim());
  if (overlayInput) {
    localStorage.setItem("ddl69_overlay_url", overlayInput.value.trim());
  }
  refreshAll();
});

const chips = document.querySelectorAll(".chip");
chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    chips.forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
  });
});

refreshAll();
