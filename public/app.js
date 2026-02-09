const watchlistUrlInput = document.getElementById("watchlistUrl");
const newsUrlInput = document.getElementById("newsUrl");
const loadBtn = document.getElementById("loadBtn");
const loadStatus = document.getElementById("loadStatus");
const watchlistCards = document.getElementById("watchlistCards");
const newsCards = document.getElementById("newsCards");
const watchlistMeta = document.getElementById("watchlistMeta");
const newsMeta = document.getElementById("newsMeta");
const startDateInput = document.getElementById("startDate");
const endDateInput = document.getElementById("endDate");
const timeframeSelect = document.getElementById("timeframe");
const segmentSelect = document.getElementById("segmentSelect");
const minProbInput = document.getElementById("minProb");
const minProbLabel = document.getElementById("minProbLabel");
const sortBySelect = document.getElementById("sortBy");
const segmentChips = document.getElementById("segmentChips");
const segmentHint = document.getElementById("segmentHint");

const statTotal = document.getElementById("statTotal");
const statAvg = document.getElementById("statAvg");
const statTop = document.getElementById("statTop");
const statTopMeta = document.getElementById("statTopMeta");
const statUpdated = document.getElementById("statUpdated");
const statWindow = document.getElementById("statWindow");
const statNews = document.getElementById("statNews");

const detailSymbol = document.getElementById("detailSymbol");
const detailScore = document.getElementById("detailScore");
const detailProb = document.getElementById("detailProb");
const detailLabel = document.getElementById("detailLabel");
const detailEvent = document.getElementById("detailEvent");
const detailWeights = document.getElementById("detailWeights");
const detailChart = document.getElementById("detailChart");
const detailNews = document.getElementById("detailNews");
const detailSparkline = document.getElementById("detailSparkline");
const detailPanel = document.getElementById("detailPanel");
const detailZone = document.getElementById("detailZone");

const DEFAULT_WATCHLIST =
  "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/watchlist/watchlist_2026-02-09.json";
const DEFAULT_NEWS =
  "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/news/polygon_news_2026-02-08.json";

let latestNews = [];

function withCacheBust(url) {
  if (!url) return url;
  const bust = `cb=${Date.now()}`;
  return url.includes("?") ? `${url}&${bust}` : `${url}?${bust}`;
}

function parseDate(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function computeRange() {
  const end = parseDate(endDateInput.value) || new Date();
  let start = parseDate(startDateInput.value);
  if (!start) {
    const tf = timeframeSelect.value;
    const days = tf === "1w" ? 7 : tf === "1m" ? 30 : tf === "3m" ? 90 : tf === "6m" ? 180 : 1;
    start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
  }
  return { start, end };
}

function pct(value) {
  const v = Number(value || 0);
  return `${(v * 100).toFixed(1)}%`;
}

function getTimeframeKey() {
  const tf = timeframeSelect ? timeframeSelect.value : "1d";
  if (tf === "1w") return "1w";
  if (tf === "1m") return "1m";
  if (tf === "3m") return "3m";
  if (tf === "6m") return "6m";
  return "1d";
}

function getTimeframeProb(row) {
  const tfKey = getTimeframeKey();
  if (row && row.timeframe_probs && row.timeframe_probs[tfKey] != null) {
    return Number(row.timeframe_probs[tfKey]);
  }
  return Number(row.p_accept || 0);
}

function historyKey(ticker) {
  return `ddl69_hist_${ticker}`;
}

function updateHistory(ticker, value) {
  if (!ticker) return [];
  const key = historyKey(ticker);
  const existing = JSON.parse(localStorage.getItem(key) || "[]");
  if (existing.length === 0 || existing[existing.length - 1] !== value) {
    existing.push(value);
  }
  const trimmed = existing.slice(-30);
  localStorage.setItem(key, JSON.stringify(trimmed));
  return trimmed;
}

function getHistory(ticker) {
  if (!ticker) return [];
  return JSON.parse(localStorage.getItem(historyKey(ticker)) || "[]");
}

function buildWeightsHtml(weights) {
  if (!weights || Object.keys(weights).length === 0) {
    return "<div class=\"helper\">No weights available.</div>";
  }
  const evidenceKeys = new Set([
    "SIGNALS_ROWS",
    "NEWS_SENTIMENT",
    "EVENT_PROXIMITY",
    "SOCIAL_SENTIMENT",
    "QLIB_SIGNAL",
    "MC_PROB",
    "MC_RETURN_PROB",
    "LOPEZ_BARRIER",
    "OPTIONS_FLOW",
    "FINVIZ_POPULAR",
    "DIRECTION_ENGINE",
    "REGIME_HMM",
    "TA_COMPOSITE"
  ]);
  const entries = Object.entries(weights).map(([k, v]) => [k, Number(v)]);
  const evidence = entries.filter(([k]) => evidenceKeys.has(k));
  const rules = entries.filter(([k]) => !evidenceKeys.has(k));
  const evidenceHtml = evidence
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<div class=\"meta\" title=\"${k}: ${pct(v)}\"><span>${k}</span><span>${pct(v)}</span></div>`)
    .join("");
  const rulesHtml = rules
    .sort((a, b) => b[1] - a[1])
    .slice(0, 16)
    .map(([k, v]) => `<div class=\"meta\" title=\"${k}: ${pct(v)}\"><span>${k}</span><span>${pct(v)}</span></div>`)
    .join("");
  return `
    <div class=\"helper\">Evidence Weights</div>
    ${evidenceHtml || "<div class=\"helper\">No evidence weights.</div>"}
    <div class=\"helper\" style=\"margin-top:6px;\">Rule Weights (top)</div>
    ${rulesHtml || "<div class=\"helper\">No rule weights.</div>"}
  `;
}

function renderChart(symbol) {
  if (!detailChart) return;
  const tf = timeframeSelect.value;
  const interval =
    tf === "1w" ? "W" :
    tf === "1m" ? "M" :
    tf === "3m" ? "W" :
    tf === "6m" ? "W" : "D";
  const height = window.innerWidth < 820 ? 380 : 520;
  detailChart.innerHTML = `
    <iframe
      src=\"https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=${interval}&theme=dark&style=1&locale=en&toolbarbg=%230b0d12&hide_side_toolbar=1&withdateranges=1&allow_symbol_change=0\"
      width=\"100%\"
      height=\"${height}\"
      loading=\"lazy\"
    ></iframe>
  `;
}

function renderSparkline(values) {
  if (!detailSparkline) return;
  const points = values.length ? values : [0];
  const max = Math.max(...points, 1);
  const min = Math.min(...points, 0);
  const range = max - min || 1;
  const w = 180;
  const h = 52;
  const step = points.length > 1 ? w / (points.length - 1) : w;
  const coords = points.map((v, i) => {
    const x = i * step;
    const y = h - ((v - min) / range) * (h - 6) - 3;
    return [x, y];
  });
  const path = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(" ");
  const fillPath = `${path} L ${w},${h} L 0,${h} Z`;
  detailSparkline.innerHTML = `
    <path class=\"sparkline-fill\" d=\"${fillPath}\" />
    <path class=\"sparkline-path\" d=\"${path}\" />
  `;
}

function filterNewsForSymbol(symbol) {
  if (!symbol || !latestNews.length) return [];
  const upper = symbol.toUpperCase();
  return latestNews.filter((item) => {
    const tickers = item.tickers || item.symbols || [];
    const title = (item.title || item.headline || "").toUpperCase();
    if (Array.isArray(tickers) && tickers.map(String).map((t) => t.toUpperCase()).includes(upper)) return true;
    return title.includes(upper);
  });
}

function renderDetailNews(symbol) {
  if (!detailNews) return;
  const items = filterNewsForSymbol(symbol).slice(0, 4);
  if (!items.length) {
    detailNews.innerHTML = "<div class=\"helper\">No symbol-specific headlines.</div>";
    return;
  }
  detailNews.innerHTML = items
    .map((item) => {
      const title = item.title || item.headline || "Untitled";
      const ts = item.published_utc || item.timestamp || item.date || "";
      const url = item.article_url || item.url || "";
      const sentiment = item.sentiment || item.sentiment_score || item.score;
      return `
        <div class=\"news-item\">
          <div class=\"news-meta\"><span>${ts}</span><span>${sentiment !== undefined ? Number(sentiment).toFixed(2) : "--"}</span></div>
          <div>${title}</div>
          ${url ? `<a href=\"${url}\" target=\"_blank\" rel=\"noopener\">Open source</a>` : ""}
        </div>
      `;
    })
    .join("");
}

function renderDetail(row) {
  if (!row) return;
  const symbol = row.ticker || row.symbol || "--";
  const event = row.event || {};
  const zone = event.zone || {};
  const currentPrice = event.current_price;
  const tfProb = getTimeframeProb(row);
  detailSymbol.textContent = symbol;
  detailScore.textContent = `${pct(row.score || 0)} score`;
  detailProb.textContent = `${pct(tfProb)} accept (${getTimeframeKey().toUpperCase()})`;
  detailLabel.textContent = row.label || "--";
  if (detailEvent) {
    const horizon = event.horizon ? `${event.horizon.value || ""}${event.horizon.type || ""}` : "--";
    const zoneText = zone && (zone.low || zone.high)
      ? `zone ${zone.low ?? "--"} → ${zone.high ?? "--"}`
      : "zone --";
    const targetVal = event.effective_target ?? event.target_price;
    const targetText = targetVal ? `target ${targetVal}` : "target --";
    const exp = event.expected_return != null ? `exp ${pct(event.expected_return)}` : "exp --";
    const rr = event.risk_reward != null ? `rr ${Number(event.risk_reward).toFixed(2)}` : "rr --";
    const cond = row.conditional && row.conditional.if_accept_then_follow != null
      ? `if accept → ${pct(row.conditional.if_accept_then_follow)}`
      : "if accept → --";
    const targetHit = event.target_hit_prob != null ? `target hit ${pct(event.target_hit_prob)}` : "target hit --";
    const reject = row.p_reject != null ? `reject ${pct(row.p_reject)}` : "reject --";
    const breakFail = row.p_break_fail != null ? `break/fail ${pct(row.p_break_fail)}` : "break/fail --";
    const confirm = event.confirm_required ? "confirm: yes" : "confirm: no";
    const confLine = row.confidence != null ? `confidence ${pct(row.confidence)}` : "confidence --";
    detailEvent.textContent = `${event.event_type || "ZONE_ACCEPT"} | ${horizon} | ${zoneText} | ${targetText} | ${exp} | ${rr} | ${cond} | ${targetHit} | ${reject} | ${breakFail} | ${confirm} | ${confLine}`;
  }
  detailWeights.innerHTML = buildWeightsHtml(row.weights || row.weights_json || {});
  renderChart(symbol);
  const history = updateHistory(symbol, Number(tfProb || 0));
  renderSparkline(history);
  renderDetailNews(symbol);
  renderZoneVisual(zone, currentPrice);
  if (detailLabel) {
    const horizon = row.horizon || {};
    const hText = horizon.horizon ? `${horizon.horizon}` : "--";
    const mcap = row.market_cap_bucket ? row.market_cap_bucket.toUpperCase() : "--";
    const qlib = row.qlib_score != null ? Number(row.qlib_score).toFixed(3) : "--";
    detailLabel.textContent = `${row.label || "--"} | horizon ${hText} | cap ${mcap} | qlib ${qlib}`;
  }
  if (detailChart) {
    detailChart.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderZoneVisual(zone, currentPrice) {
  if (!detailZone) return;
  const low = zone?.low;
  const high = zone?.high;
  if (low == null || high == null || !Number.isFinite(Number(low)) || !Number.isFinite(Number(high))) {
    detailZone.innerHTML = "<div class=\"helper\">No zone data.</div>";
    return;
  }
  const min = Math.min(Number(low), Number(high));
  const max = Math.max(Number(low), Number(high));
  const price = Number.isFinite(Number(currentPrice)) ? Number(currentPrice) : null;
  const range = max - min || 1;
  const pricePos = price == null ? null : Math.min(100, Math.max(0, ((price - min) / range) * 100));
  detailZone.innerHTML = `
    <div class="zone-track">
      <div class="zone-range" style="left:0%; width:100%"></div>
      ${pricePos == null ? "" : `<div class="zone-marker" style="left:${pricePos}%"></div>`}
      <div class="zone-label zone-low">${min.toFixed(2)}</div>
      <div class="zone-label zone-high">${max.toFixed(2)}</div>
      ${pricePos == null ? "" : `<div class="zone-label zone-price" style="left:${pricePos}%">${price.toFixed(2)}</div>`}
    </div>
  `;
}

function renderWatchlist(data) {
  if (!data || !Array.isArray(data.ranked)) {
    watchlistCards.innerHTML = "<div class=\"helper\">No watchlist data.</div>";
    watchlistMeta.textContent = "No data loaded";
    statTotal.textContent = "0";
    statAvg.textContent = "0%";
    statTop.textContent = "--";
    statTopMeta.textContent = "No data";
    return [];
  }

  const { start, end } = computeRange();
  const minProb = parseFloat(minProbInput.value || "0");
  const segment = segmentSelect ? segmentSelect.value : "all";
  const segmentRows =
    segment !== "all" && data.segments && Array.isArray(data.segments[segment])
      ? data.segments[segment]
      : data.ranked;

  const filtered = segmentRows.filter((row) => {
    const p = getTimeframeProb(row);
    if (typeof p === "number" && p < minProb) return false;
    if (!row.created_at) return true;
    const created = new Date(row.created_at);
    if (Number.isNaN(created.getTime())) return true;
    return created >= start && created <= end;
  });
  const sortBy = sortBySelect ? sortBySelect.value : "prob";
  const sorted = filtered.slice().sort((a, b) => {
    if (sortBy === "return") {
      return Number((b.event && b.event.expected_return) || -1e9) - Number((a.event && a.event.expected_return) || -1e9);
    }
    if (sortBy === "rr") {
      return Number((b.event && b.event.risk_reward) || -1e9) - Number((a.event && a.event.risk_reward) || -1e9);
    }
    if (sortBy === "score") {
      return Number(b.score || 0) - Number(a.score || 0);
    }
    return getTimeframeProb(b) - getTimeframeProb(a);
  });

  const segmentLabel = segment === "all" ? "All" : segment.replace("_", " ");
  watchlistMeta.textContent = `As of ${data.asof || "unknown"} - ${sorted.length} ideas (${segmentLabel})`;
  renderSegmentChips(data);
  statUpdated.textContent = data.asof ? new Date(data.asof).toLocaleString() : "--";
  statTotal.textContent = String(sorted.length);
  statWindow.textContent = `${timeframeSelect.value.toUpperCase()} window`;

  if (sorted.length > 0) {
    const avg = sorted.reduce((acc, row) => acc + Number(getTimeframeProb(row) || 0), 0) / sorted.length;
    statAvg.textContent = pct(avg);
    statTop.textContent = sorted[0].ticker || "--";
    statTopMeta.textContent = `${pct(getTimeframeProb(sorted[0]))} - ${sorted[0].label || "signal"}`;
    renderDetail(sorted[0]);
  } else {
    statAvg.textContent = "0%";
    statTop.textContent = "--";
    statTopMeta.textContent = "No data";
  }

  watchlistCards.innerHTML = sorted
    .map((row, idx) => {
      const score = Number(row.score || 0).toFixed(3);
      const pAccept = Number(getTimeframeProb(row) || 0).toFixed(3);
      const weightKeys = row.weights ? Object.keys(row.weights).slice(0, 3).join(" - ") : "";
      const exp = row.event && row.event.expected_return != null ? pct(row.event.expected_return) : "--";
      const rr = row.event && row.event.risk_reward != null ? Number(row.event.risk_reward).toFixed(2) : "--";
      const newsItems = filterNewsForSymbol(row.ticker);
      const newsCount = newsItems.length;
      const headline = newsCount ? (newsItems[0].title || newsItems[0].headline || "").slice(0, 80) : "";
      return `
      <article class=\"card\" data-idx=\"${idx}\">
        <div class=\"card-top\">
          <div>
            <div class=\"ticker\">${row.ticker}</div>
            <div class=\"helper\">${row.plan_type || "watchlist"}</div>
          </div>
          <span class=\"pill\">${row.label || "signal"}</span>
        </div>
        <div class=\"probability\">
          <div class=\"helper\">Probability of Accept</div>
          <span>${pct(getTimeframeProb(row))}</span>
          <div class=\"bar\"><div class=\"bar-fill\" style=\"width:${Math.min(100, getTimeframeProb(row) * 100)}%\"></div></div>
        </div>
        <div class=\"meta\">
          <span>Score</span>
          <span>${score}</span>
        </div>
        <div class=\"meta\">
          <span>Expected Return</span>
          <span>${exp}</span>
        </div>
        <div class=\"meta\">
          <span>Risk/Reward</span>
          <span>${rr}</span>
        </div>
        <div class=\"meta\">
          <span>News</span>
          <span>${newsCount}</span>
        </div>
        ${headline ? `<div class=\"helper\">${headline}...</div>` : ""}
        <div class=\"helper\">${weightKeys}</div>
        <div class=\"helper\">p_accept(${getTimeframeKey().toUpperCase()}) ${pAccept}</div>
      </article>`;
    })
    .join("");

  watchlistCards.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => {
      const idx = Number(card.dataset.idx || 0);
      renderDetail(sorted[idx]);
      if (detailPanel) {
        detailPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  return sorted;
}

function renderSegmentChips(data) {
  if (!segmentChips) return;
  const segments = data && data.segments ? data.segments : {};
  const entries = [
    { key: "all", label: "All", count: Array.isArray(data?.ranked) ? data.ranked.length : 0 },
    { key: "sp500", label: "S&P 500", count: (segments.sp500 || []).length },
    { key: "large_cap", label: "Large", count: (segments.large_cap || []).length },
    { key: "mid_cap", label: "Mid", count: (segments.mid_cap || []).length },
    { key: "small_cap", label: "Small", count: (segments.small_cap || []).length },
    { key: "social_trending", label: "Social", count: (segments.social_trending || []).length },
  ];
  const current = segmentSelect ? segmentSelect.value : "all";
  segmentChips.innerHTML = entries
    .map((e) => {
      const active = e.key === current ? "active" : "";
      return `<button class="chip ${active}" data-seg="${e.key}">${e.label} · ${e.count}</button>`;
    })
    .join("");
  segmentChips.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (!segmentSelect) return;
      segmentSelect.value = chip.dataset.seg || "all";
      loadData().catch(() => {});
    });
  });
  if (segmentHint) {
    const missingCaps = (segments.small_cap || []).length === 0 && (segments.mid_cap || []).length === 0;
    segmentHint.textContent = missingCaps
      ? "Mid/Small caps are empty until market-cap data is populated."
      : "Segments depend on available market-cap + universe data.";
  }
}

function renderNews(items) {
  if (!Array.isArray(items) || items.length === 0) {
    newsCards.innerHTML = "<div class=\"helper\">No news data.</div>";
    newsMeta.textContent = "No data loaded";
    statNews.textContent = "0";
    return;
  }

  const scored = items.map((item) => {
    const sentiment = Number(item.sentiment || item.sentiment_score || 0);
    const ts = new Date(item.published_utc || item.published_at || item.updated_at || Date.now());
    const ageH = Math.max(1, (Date.now() - ts.getTime()) / 3600000);
    const score = Math.abs(sentiment) / Math.sqrt(ageH);
    return { ...item, _score: score };
  });
  const top = scored.sort((a, b) => b._score - a._score).slice(0, 8);
  newsMeta.textContent = `${top.length} weighted headlines`;
  statNews.textContent = String(top.length);

  newsCards.innerHTML = top
    .map((item) => {
      const published = item.published_utc || item.published_at || item.updated_at || "";
      const url = item.article_url || item.url || "";
      const publisher = item.publisher?.name || item.publisher || "";
      return `
        <article class=\"news-item\">
          <div class=\"helper\">${published}</div>
          <div class=\"ticker\">${item.title || "Untitled"}</div>
          <div class=\"helper\">${item.description || ""}</div>
          <div class=\"meta\">
            <span>${publisher}</span>
            ${url ? `<a href=\"${url}\" target=\"_blank\" rel=\"noopener\">Open</a>` : ""}
          </div>
        </article>`;
    })
    .join("");
}

async function loadData() {
  const watchlistUrl = watchlistUrlInput.value.trim();
  const newsUrl = newsUrlInput.value.trim();
  loadStatus.textContent = "Loading...";

  try {
    let watchlistData = null;
    if (watchlistUrl) {
      const res = await fetch(withCacheBust(watchlistUrl), { cache: "no-store" });
      watchlistData = await res.json();
    }

    if (newsUrl) {
      const res = await fetch(withCacheBust(newsUrl), { cache: "no-store" });
      const data = await res.json();
      latestNews = data.results || data.items || data || [];
      renderNews(latestNews);
    }

    const filtered = renderWatchlist(watchlistData);

    if (filtered.length === 0) {
      loadStatus.textContent = "Loaded, no matches in filter.";
    } else {
      loadStatus.textContent = "Loaded.";
    }
  } catch (err) {
    console.error(err);
    loadStatus.textContent = "Load failed.";
  }
}

minProbInput.addEventListener("input", () => {
  minProbLabel.textContent = Number(minProbInput.value).toFixed(2);
});

[timeframeSelect, startDateInput, endDateInput, minProbInput, sortBySelect, segmentSelect].forEach((el) => {
  if (!el) return;
  el.addEventListener("change", () => loadData().catch(() => {}));
});

loadBtn.addEventListener("click", () => {
  loadData().catch(() => {});
});

if (!watchlistUrlInput.value) watchlistUrlInput.value = DEFAULT_WATCHLIST;
if (!newsUrlInput.value) newsUrlInput.value = DEFAULT_NEWS;

minProbLabel.textContent = Number(minProbInput.value).toFixed(2);
loadData().catch(() => {});
setInterval(() => loadData().catch(() => {}), 5 * 60 * 1000);

