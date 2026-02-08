const watchlistUrlInput = document.getElementById("watchlistUrl");
const newsUrlInput = document.getElementById("newsUrl");
const loadBtn = document.getElementById("loadBtn");
const resetBtn = document.getElementById("resetBtn");
const watchlistCards = document.getElementById("watchlistCards");
const newsCards = document.getElementById("newsCards");
const watchlistMeta = document.getElementById("watchlistMeta");
const newsMeta = document.getElementById("newsMeta");
const mosaic = document.getElementById("mosaic");
const mosaicMeta = document.getElementById("mosaicMeta");
const startDateInput = document.getElementById("startDate");
const endDateInput = document.getElementById("endDate");
const timeframeSelect = document.getElementById("timeframe");
const minProbInput = document.getElementById("minProb");
const minProbLabel = document.getElementById("minProbLabel");
const scopeValue = document.getElementById("scopeValue");
const timeframeLabel = document.getElementById("timeframeLabel");
const lastUpdate = document.getElementById("lastUpdate");
const filterHint = document.getElementById("filterHint");
const scopeChips = document.getElementById("scopeChips");

const DEFAULT_WATCHLIST_URL =
  "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/watchlist/watchlist_2026-02-08.json";
const DEFAULT_NEWS_URL =
  "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/news/polygon_news_2026-02-08.json";
const AUTO_REFRESH_MS = 5 * 60 * 1000;

const scopePresets = {
  day: { label: "Day Trade", timeframe: "1d", minProb: 0.35 },
  swing: { label: "Swing", timeframe: "1w", minProb: 0.45 },
  long: { label: "Long", timeframe: "1m", minProb: 0.55 },
};

let currentScope = "day";

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
    const days = tf === "1w" ? 7 : tf === "1m" ? 30 : tf === "3m" ? 90 : 1;
    start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
  }
  return { start, end };
}

function pct(value) {
  const v = Number(value || 0);
  return `${(v * 100).toFixed(1)}%`;
}

function updateSummary() {
  minProbLabel.textContent = Number(minProbInput.value).toFixed(2);
  timeframeLabel.textContent = timeframeSelect.value.toUpperCase();
  scopeValue.textContent = scopePresets[currentScope].label;
}

function renderWatchlist(data) {
  if (!data || !Array.isArray(data.ranked)) {
    watchlistCards.innerHTML = "<div class=\"muted\">No watchlist data.</div>";
    return;
  }
  const { start, end } = computeRange();
  const minProb = parseFloat(minProbInput.value || "0");

  const filtered = data.ranked.filter((row) => {
    if (typeof row.p_accept === "number" && row.p_accept < minProb) return false;
    if (!row.created_at) return true;
    const created = new Date(row.created_at);
    if (Number.isNaN(created.getTime())) return true;
    return created >= start && created <= end;
  });

  watchlistMeta.textContent = `As of ${data.asof || "unknown"} 5 ${filtered.length} ideas`;
  filterHint.textContent = `Filters active: ${minProb > 0 ? 1 : 0}`;

  watchlistCards.innerHTML = filtered
    .map((row) => {
      const score = Number(row.score || 0).toFixed(3);
      const pAccept = Number(row.p_accept || 0).toFixed(3);
      const weightSummary = row.weights ? Object.keys(row.weights).slice(0, 2).join(" 7 ") : "";

      return `
      <article class="card">
        <div class="row" style="justify-content: space-between; align-items:center;">
          <div>
            <div class="ticker">${row.ticker}</div>
            <div class="muted" style="font-size:0.75rem;">${row.plan_type || ""}</div>
          </div>
          <span class="pill">${row.label || "signal"}</span>
        </div>
        <div>
          <div class="muted" style="font-size:0.75rem;">Probability of Accept</div>
          <div style="font-size:1.6rem; font-weight:600;">${pct(row.p_accept)}</div>
          <div class="progress"><span style="width:${Math.min(100, row.p_accept * 100)}%"></span></div>
        </div>
        <div class="row" style="justify-content: space-between;">
          <div class="muted">Score</div>
          <div style="font-weight:600;">${score}</div>
        </div>
        <div class="muted" style="font-size:0.75rem;">${weightSummary}</div>
        <div class="muted" style="font-size:0.75rem;">p_accept ${pAccept}</div>
      </article>`;
    })
    .join("");

  renderMosaic(filtered);
}

function renderMosaic(rows) {
  if (!rows || rows.length === 0) {
    mosaic.innerHTML = "<div class=\"muted\">No signals yet.</div>";
    mosaicMeta.textContent = "Awaiting data";
    return;
  }
  const top = rows.slice(0, 6);
  mosaicMeta.textContent = `${top.length} tickers highlighted`;
  mosaic.innerHTML = top
    .map((row) => {
      return `
      <div class="mosaic-tile">
        <div class="ticker">${row.ticker}</div>
        <div class="muted" style="font-size:0.75rem;">${row.label || "signal"}</div>
        <div style="margin-top:12px; font-size:1.2rem; font-weight:600;">${pct(row.p_accept)}</div>
      </div>`;
    })
    .join("");
}

function renderNews(items) {
  if (!Array.isArray(items) || items.length === 0) {
    newsCards.innerHTML = "<div class=\"muted\">No news data.</div>";
    return;
  }
  const top = items.slice(0, 8);
  newsMeta.textContent = `${top.length} headlines`;
  newsCards.innerHTML = top
    .map((item) => {
      const published = item.published_utc || item.published_at || item.updated_at || "";
      const url = item.article_url || item.url || "";
      const publisher = item.publisher?.name || item.publisher || "";
      return `
        <article class="news-card">
          <div class="muted" style="font-size:0.7rem;">${published}</div>
          <div style="font-weight:600;">${item.title || "Untitled"}</div>
          <div class="muted" style="font-size:0.85rem;">${item.description || ""}</div>
          <div class="row" style="justify-content: space-between;">
            <span class="muted">${publisher}</span>
            ${url ? `<a href="${url}" target="_blank" rel="noopener">Open</a>` : ""}
          </div>
        </article>`;
    })
    .join("");
}

async function loadData() {
  const watchlistUrl = watchlistUrlInput.value.trim();
  const newsUrl = newsUrlInput.value.trim();

  if (watchlistUrl) {
    const res = await fetch(withCacheBust(watchlistUrl));
    const data = await res.json();
    renderWatchlist(data);
  }

  if (newsUrl) {
    const res = await fetch(withCacheBust(newsUrl));
    const data = await res.json();
    renderNews(data);
  }

  lastUpdate.textContent = `Updated ${new Date().toLocaleString()}`;
}

function applyScope(scope) {
  currentScope = scope;
  const preset = scopePresets[scope];
  minProbInput.value = preset.minProb;
  timeframeSelect.value = preset.timeframe;
  document.querySelectorAll(".chip").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.scope === scope);
  });
  updateSummary();
}

scopeChips.addEventListener("click", (event) => {
  const btn = event.target.closest(".chip");
  if (!btn) return;
  applyScope(btn.dataset.scope);
  loadData().catch(() => {});
});

[minProbInput, timeframeSelect, startDateInput, endDateInput].forEach((el) => {
  el.addEventListener("change", () => {
    updateSummary();
    loadData().catch(() => {});
  });
});

loadBtn.addEventListener("click", () => {
  loadData().catch((err) => console.error(err));
});

resetBtn.addEventListener("click", () => {
  startDateInput.value = "";
  endDateInput.value = "";
  applyScope("day");
  loadData().catch(() => {});
});

if (!watchlistUrlInput.value) watchlistUrlInput.value = DEFAULT_WATCHLIST_URL;
if (!newsUrlInput.value) newsUrlInput.value = DEFAULT_NEWS_URL;

applyScope("day");
updateSummary();
loadData().catch(() => {});
setInterval(() => loadData().catch(() => {}), AUTO_REFRESH_MS);
