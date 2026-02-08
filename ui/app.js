const DEFAULT_WATCHLIST = "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/watchlist/watchlist_2026-02-08.json";
const DEFAULT_NEWS = "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/news/polygon_news_2026-02-08.json";

const watchlistInput = document.getElementById("watchlistUrl");
const newsInput = document.getElementById("newsUrl");
const refreshBtn = document.getElementById("refreshBtn");
const saveBtn = document.getElementById("saveBtn");
const topNInput = document.getElementById("topN");

const asofValue = document.getElementById("asofValue");
const sourceValue = document.getElementById("sourceValue");
const countValue = document.getElementById("countValue");
const watchlistMeta = document.getElementById("watchlistMeta");
const watchlistGrid = document.getElementById("watchlistGrid");
const scoreBars = document.getElementById("scoreBars");
const newsMeta = document.getElementById("newsMeta");
const newsGrid = document.getElementById("newsGrid");

const modalSymbol = document.getElementById("modalSymbol");
const modalScore = document.getElementById("modalScore");
const modalProb = document.getElementById("modalProb");
const modalLabel = document.getElementById("modalLabel");
const modalWeights = document.getElementById("modalWeights");
const tvChart = document.getElementById("tvChart");

const storedWatchlist = localStorage.getItem("ddl69_watchlist_url") || DEFAULT_WATCHLIST;
const storedNews = localStorage.getItem("ddl69_news_url") || DEFAULT_NEWS;
watchlistInput.value = storedWatchlist;
newsInput.value = storedNews;

function formatDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toISOString().replace("T", " ").replace("Z", " UTC");
}

function scoreBar(label, value) {
  const pct = Math.max(0, Math.min(1, value || 0));
  return `
    <div>
      <div class="watch-meta"><span>${label}</span><span>${(pct * 100).toFixed(1)}%</span></div>
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
    .slice(0, 8)
    .map(
      ([k, v]) =>
        `<div class="weight-item"><span>${k}</span><span>${(Number(v) * 100).toFixed(1)}%</span></div>`
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

function openSymbolModal(row) {
  const symbol = row.ticker || row.symbol || "—";
  const score = Number(row.score || 0);
  const prob = Number(row.p_accept || 0);

  modalSymbol.textContent = symbol;
  modalScore.textContent = `${(score * 100).toFixed(1)}% score`;
  modalProb.textContent = `${(prob * 100).toFixed(1)}% accept`;
  modalLabel.textContent = row.label || "—";
  modalWeights.innerHTML = buildWeightsHtml(row.weights || row.weights_json || {});
  renderTradingView(symbol);

  if (window.UIkit) {
    UIkit.modal("#symbolModal").show();
  }
}

function renderWatchlist(data) {
  const topN = Number(topNInput.value || 10);
  watchlistGrid.innerHTML = "";
  if (!data) {
    watchlistMeta.textContent = "No watchlist data.";
    return;
  }

  const asof = data.asof || data.generated_at || "";
  asofValue.textContent = formatDate(asof);
  sourceValue.textContent = data.source || data.provider || "Supabase artifacts";

  if (Array.isArray(data.ranked) && data.ranked.length) {
    const ranked = data.ranked.slice(0, topN);
    countValue.textContent = data.ranked.length;
    watchlistMeta.textContent = `Ranked list · showing top ${ranked.length}`;
    renderScoreBars(data.ranked);

    ranked.forEach((row) => {
      const card = document.createElement("div");
      card.className = "watch-card";
      card.innerHTML = `
        <h4>${row.ticker || row.symbol}</h4>
        <div class="watch-meta"><span>${row.label || "—"}</span><span>Score ${(Number(row.score || 0) * 100).toFixed(1)}%</span></div>
        <div class="watch-meta"><span>P(accept)</span><span>${(Number(row.p_accept || 0) * 100).toFixed(1)}%</span></div>
        <div class="watch-meta"><span>Plan</span><span>${row.plan_type || "—"}</span></div>
        <div class="weight-list">${buildWeightsHtml(row.weights || row.weights_json || {})}</div>
      `;
      card.addEventListener("click", () => openSymbolModal(row));
      watchlistGrid.appendChild(card);
    });
    return;
  }

  if (Array.isArray(data.tickers)) {
    const tickers = data.tickers.slice(0, topN);
    countValue.textContent = data.count || data.tickers.length;
    watchlistMeta.textContent = `Universe list · showing ${tickers.length}`;
    renderScoreBars([]);
    tickers.forEach((t) => {
      const row = { ticker: t, label: "Universe", p_accept: 0, score: 0, weights: {} };
      const card = document.createElement("div");
      card.className = "watch-card";
      card.innerHTML = `
        <h4>${t}</h4>
        <div class="watch-meta"><span>Universe</span><span>Member</span></div>
        <div class="small-note">No ranking data in this list.</div>
      `;
      card.addEventListener("click", () => openSymbolModal(row));
      watchlistGrid.appendChild(card);
    });
    return;
  }

  watchlistMeta.textContent = "Unsupported watchlist format.";
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
    const title = item.title || item.headline || "Untitled";
    const ts = item.published_utc || item.timestamp || item.time || item.date;
    const tickers = item.tickers || item.symbols || [];
    const url = item.article_url || item.url || "#";
    const sentiment = item.sentiment || item.sentiment_score || item.score;

    const card = document.createElement("div");
    card.className = "news-card";
    card.innerHTML = `
      <h5>${title}</h5>
      <div class="news-meta">
        <span>${formatDate(ts)}</span>
        <span>${Array.isArray(tickers) ? tickers.slice(0, 3).join(", ") : tickers}</span>
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
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Fetch failed: ${resp.status}`);
  }
  return resp.json();
}

async function refreshAll() {
  try {
    const [watchlist, news] = await Promise.all([
      fetchJson(watchlistInput.value.trim()),
      fetchJson(newsInput.value.trim()),
    ]);
    renderWatchlist(watchlist);
    renderNews(news);
  } catch (err) {
    watchlistMeta.textContent = `Error: ${err.message}`;
    newsMeta.textContent = `Error: ${err.message}`;
  }
}

refreshBtn.addEventListener("click", refreshAll);

saveBtn.addEventListener("click", () => {
  localStorage.setItem("ddl69_watchlist_url", watchlistInput.value.trim());
  localStorage.setItem("ddl69_news_url", newsInput.value.trim());
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
