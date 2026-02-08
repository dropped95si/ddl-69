const watchlistUrlInput = document.getElementById("watchlistUrl");
const newsUrlInput = document.getElementById("newsUrl");
const loadBtn = document.getElementById("loadBtn");
const watchlistCards = document.getElementById("watchlistCards");
const newsCards = document.getElementById("newsCards");
const watchlistMeta = document.getElementById("watchlistMeta");
const newsMeta = document.getElementById("newsMeta");
const startDateInput = document.getElementById("startDate");
const endDateInput = document.getElementById("endDate");
const timeframeSelect = document.getElementById("timeframe");
const minProbInput = document.getElementById("minProb");
const minProbLabel = document.getElementById("minProbLabel");

const defaultImages = [
  "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=900&q=80",
  "https://images.unsplash.com/photo-1487014679447-9f8336841d58?auto=format&fit=crop&w=900&q=80",
  "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=900&q=80",
  "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=900&q=80",
];

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

function renderWatchlist(data) {
  if (!data || !Array.isArray(data.ranked)) {
    watchlistCards.innerHTML = "<div class=\"text-mist/50\">No watchlist data.</div>";
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

  watchlistMeta.textContent = `As of ${data.asof || "unknown"} • ${filtered.length} ideas`;

  watchlistCards.innerHTML = filtered
    .map((row, idx) => {
      const image = defaultImages[idx % defaultImages.length];
      const score = Number(row.score || 0).toFixed(3);
      const pAccept = Number(row.p_accept || 0).toFixed(3);
      return `
      <div class="card rounded-2xl p-5 flex flex-col gap-4 fade-in" style="animation-delay:${idx * 40}ms">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-xl overflow-hidden">
              <img src="${image}" alt="" class="h-full w-full object-cover" />
            </div>
            <div>
              <div class="text-xl font-semibold">${row.ticker}</div>
              <div class="text-xs text-mist/50">${row.plan_type || ""}</div>
            </div>
          </div>
          <span class="badge text-xs px-2 py-1 rounded-full">${row.label || "signal"}</span>
        </div>
        <div>
          <div class="text-xs text-mist/50">Probability of Accept</div>
          <div class="text-2xl font-semibold text-lime">${pct(row.p_accept)}</div>
          <div class="h-2 bg-black/40 rounded-full mt-2 overflow-hidden">
            <div class="h-full bg-lime" style="width:${Math.min(100, row.p_accept * 100)}%"></div>
          </div>
        </div>
        <div class="flex items-center justify-between text-sm">
          <div class="text-mist/60">Score</div>
          <div class="font-semibold">${score}</div>
        </div>
        <div class="text-xs text-mist/60">p_accept ${pAccept}</div>
      </div>`;
    })
    .join("");
}

function renderNews(items) {
  if (!Array.isArray(items) || items.length === 0) {
    newsCards.innerHTML = "<div class=\"text-mist/50\">No news data.</div>";
    return;
  }
  const top = items.slice(0, 10);
  newsMeta.textContent = `${top.length} headlines`;
  newsCards.innerHTML = top
    .map((item, idx) => {
      const published = item.published_utc || item.published_at || item.updated_at || "";
      const url = item.article_url || item.url || "";
      const publisher = item.publisher?.name || item.publisher || "";
      return `
        <div class="card rounded-2xl p-5 fade-in" style="animation-delay:${idx * 40}ms">
          <div class="text-xs text-mist/50">${published}</div>
          <div class="text-lg font-semibold mt-2">${item.title || "Untitled"}</div>
          <div class="text-sm text-mist/70 mt-2">${item.description || ""}</div>
          <div class="flex items-center justify-between mt-4 text-xs text-mist/50">
            <span>${publisher}</span>
            ${url ? `<a class="text-ember" href="${url}" target="_blank" rel="noopener">Open</a>` : ""}
          </div>
        </div>`;
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
}

minProbInput.addEventListener("input", () => {
  minProbLabel.textContent = Number(minProbInput.value).toFixed(2);
});

[timeframeSelect, startDateInput, endDateInput, minProbInput].forEach((el) => {
  el.addEventListener("change", () => loadData().catch(() => {}));
});

loadBtn.addEventListener("click", () => {
  loadData().catch((err) => {
    console.error(err);
  });
});

minProbLabel.textContent = Number(minProbInput.value).toFixed(2);
