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
const minProbInput = document.getElementById("minProb");
const minProbLabel = document.getElementById("minProbLabel");

const statTotal = document.getElementById("statTotal");
const statAvg = document.getElementById("statAvg");
const statTop = document.getElementById("statTop");
const statTopMeta = document.getElementById("statTopMeta");
const statUpdated = document.getElementById("statUpdated");
const statWindow = document.getElementById("statWindow");
const statNews = document.getElementById("statNews");

const DEFAULT_WATCHLIST =
  "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/watchlist/watchlist_2026-02-08.json";
const DEFAULT_NEWS =
  "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/news/polygon_news_2026-02-08.json";

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

function renderWatchlist(data) {
  if (!data || !Array.isArray(data.ranked)) {
    watchlistCards.innerHTML = "<div class=\"helper\">No watchlist data.</div>";
    watchlistMeta.textContent = "No data loaded";
    statTotal.textContent = "0";
    statAvg.textContent = "0%";
    statTop.textContent = "—";
    statTopMeta.textContent = "No data";
    return [];
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

  watchlistMeta.textContent = `As of ${data.asof || "unknown"} • ${filtered.length} ideas`;
  statUpdated.textContent = data.asof ? new Date(data.asof).toLocaleString() : "—";
  statTotal.textContent = String(filtered.length);
  statWindow.textContent = `${timeframeSelect.value.toUpperCase()} window`;

  if (filtered.length > 0) {
    const avg = filtered.reduce((acc, row) => acc + Number(row.p_accept || 0), 0) / filtered.length;
    statAvg.textContent = pct(avg);
    statTop.textContent = filtered[0].ticker || "—";
    statTopMeta.textContent = `${pct(filtered[0].p_accept)} • ${filtered[0].label || "signal"}`;
  } else {
    statAvg.textContent = "0%";
    statTop.textContent = "—";
    statTopMeta.textContent = "No data";
  }

  watchlistCards.innerHTML = filtered
    .map((row) => {
      const score = Number(row.score || 0).toFixed(3);
      const pAccept = Number(row.p_accept || 0).toFixed(3);
      const weightKeys = row.weights ? Object.keys(row.weights).slice(0, 3).join(" • ") : "";
      return `
      <article class="card">
        <div class="card-top">
          <div>
            <div class="ticker">${row.ticker}</div>
            <div class="helper">${row.plan_type || "watchlist"}</div>
          </div>
          <span class="pill">${row.label || "signal"}</span>
        </div>
        <div class="probability">
          <div class="helper">Probability of Accept</div>
          <span>${pct(row.p_accept)}</span>
          <div class="bar"><div class="bar-fill" style="width:${Math.min(100, row.p_accept * 100)}%"></div></div>
        </div>
        <div class="meta">
          <span>Score</span>
          <span>${score}</span>
        </div>
        <div class="helper">${weightKeys}</div>
        <div class="helper">p_accept ${pAccept}</div>
      </article>`;
    })
    .join("");

  return filtered;
}

function renderNews(items) {
  if (!Array.isArray(items) || items.length === 0) {
    newsCards.innerHTML = "<div class=\"helper\">No news data.</div>";
    newsMeta.textContent = "No data loaded";
    statNews.textContent = "0";
    return;
  }

  const top = items.slice(0, 12);
  newsMeta.textContent = `${top.length} headlines`;
  statNews.textContent = String(top.length);

  newsCards.innerHTML = top
    .map((item) => {
      const published = item.published_utc || item.published_at || item.updated_at || "";
      const url = item.article_url || item.url || "";
      const publisher = item.publisher?.name || item.publisher || "";
      return `
        <article class="news-item">
          <div class="helper">${published}</div>
          <div class="ticker">${item.title || "Untitled"}</div>
          <div class="helper">${item.description || ""}</div>
          <div class="meta">
            <span>${publisher}</span>
            ${url ? `<a href="${url}" target="_blank" rel="noopener">Open</a>` : ""}
          </div>
        </article>`;
    })
    .join("");
}

async function loadData() {
  const watchlistUrl = watchlistUrlInput.value.trim();
  const newsUrl = newsUrlInput.value.trim();
  loadStatus.textContent = "Loading…";

  try {
    let watchlistData = null;
    if (watchlistUrl) {
      const res = await fetch(withCacheBust(watchlistUrl));
      watchlistData = await res.json();
    }
    const filtered = renderWatchlist(watchlistData);

    if (newsUrl) {
      const res = await fetch(withCacheBust(newsUrl));
      const data = await res.json();
      renderNews(data.results || data.items || data);
    }

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

[timeframeSelect, startDateInput, endDateInput, minProbInput].forEach((el) => {
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
