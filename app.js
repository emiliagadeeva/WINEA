// app.js (ES-module)

// 1) Импортируем Transformers.js из CDN (реальная модель, не заглушка)
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3';

// Запрещаем искать модели локально на сервере — только с Hugging Face Hub
env.allowLocalModels = false;

// 2) Глобальные данные
let wines = [];       // строки из df.csv
let embeddings = [];  // эмбеддинги вин из wine_embeddings.json
let embeddingDim = 0;

// Ленивый запуск пайплайна эмбеддингов
let embedderPromise = null;

// Удобное логирование
function log(...args) {
  console.log("[WineApp]", ...args);
}

// --- Инициализация пайплайна эмбеддингов --- //
async function getEmbedder() {
  if (!embedderPromise) {
    log("Инициализируем модель эмбеддингов Xenova/paraphrase-multilingual-MiniLM-L12-v2...");
    embedderPromise = pipeline(
      'feature-extraction',
      'Xenova/paraphrase-multilingual-MiniLM-L12-v2'
    );
  }
  return embedderPromise;
}

// --- Эмбеддинг пользовательского запроса (РЕАЛЬНЫЙ, через модель) --- //
async function getQueryEmbedding(queryText) {
  const embedder = await getEmbedder();
  const output = await embedder(queryText, { pooling: 'mean', normalize: true });

  // output — Tensor, преобразуем в обычный массив
  const arr = output.tolist(); // [[...]]
  return arr[0];
}

// --- Загрузка CSV и JSON с Google Drive --- //
async function loadData() {
  const loaderEl = document.getElementById("loader");
  const errorEl = document.getElementById("error");
  const mainEl = document.getElementById("main");

  try {
    loaderEl.hidden = false;
    errorEl.hidden = true;
    mainEl.hidden = true;

    if (!window.CSV_URL || !window.EMBEDDINGS_URL) {
      throw new Error("Не заданы CSV_URL / EMBEDDINGS_URL в config.js");
    }

    // Параллельно грузим CSV и JSON
    const [csvText, embJson] = await Promise.all([
      fetch(window.CSV_URL).then((res) => {
        if (!res.ok) throw new Error("Не удалось загрузить df.csv");
        return res.text();
      }),
      fetch(window.EMBEDDINGS_URL).then((res) => {
        if (!res.ok) throw new Error("Не удалось загрузить wine_embeddings.json");
        return res.json();
      }),
    ]);

    // Парсим CSV → wines[]
    const parsed = Papa.parse(csvText, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    });

    wines = parsed.data.map((row, index) => ({
      ...row,
      _index: index, // связь с embeddings
    }));
    log("Загружено вин:", wines.length);

    // Загружаем эмбеддинги
    embeddings = embJson.embeddings;
    embeddingDim = embJson.dimension;
    log("Загружено эмбеддингов:", embeddings.length, "размерность:", embeddingDim);

    if (!embeddings || embeddings.length === 0) {
      throw new Error("В wine_embeddings.json нет эмбеддингов");
    }

    if (embeddings.length !== wines.length) {
      console.warn(
        "⚠️ Количество эмбеддингов не совпадает с количеством строк в df.csv. " +
        "Убедись, что оба файла были созданы из одного и того же датафрейма."
      );
    }

    // Инициализируем UI
    initTabs();
    initFilters();
    initFavoritesList();
    initUseCases();

    loaderEl.hidden = true;
    mainEl.hidden = false;
  } catch (err) {
    console.error(err);
    loaderEl.hidden = true;
    errorEl.hidden = false;
    errorEl.textContent = "Ошибка при загрузке данных: " + err.message;
  }
}

// --- Косинусное сходство --- //
function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = vecA.length;

  for (let i = 0; i < len; i++) {
    const a = vecA[i];
    const b = vecB[i];
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }

  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// --- TOP-K похожих вин по эмбеддингу запроса --- //
function getTopKSimilar(queryEmbedding, topK = (window.DEFAULT_TOP_K || 10), options = {}) {
  const {
    filterCountry = "",
    filterVariety = "",
    maxPrice = null,
    excludeIndices = new Set(),
  } = options;

  const scores = [];

  for (let i = 0; i < embeddings.length; i++) {
    if (excludeIndices.has(i)) continue;

    const wine = wines[i];
    if (!wine) continue;

    // Фильтр по стране
    if (filterCountry && wine.country) {
      if (String(wine.country).toLowerCase() !== filterCountry.toLowerCase()) {
        continue;
      }
    }

    // Фильтр по variety
    if (filterVariety && wine.variety) {
      if (String(wine.variety).toLowerCase() !== filterVariety.toLowerCase()) {
        continue;
      }
    }

    // Фильтр по цене
    if (maxPrice != null && maxPrice !== "" && wine.price != null && !isNaN(wine.price)) {
      if (Number(wine.price) > Number(maxPrice)) {
        continue;
      }
    }

    const score = cosineSimilarity(queryEmbedding, embeddings[i]);
    scores.push({ index: i, score });
  }

  scores.sort((a, b) => b.score - a.score);

  return scores.slice(0, topK).map(({ index, score }) => ({
    wine: wines[index],
    similarity: score,
  }));
}

// --- Средний эмбеддинг выбранных вин (для кейса 3) --- //
function averageEmbeddings(indices) {
  if (!indices.length) return null;
  const dim = embeddingDim || (embeddings[0] ? embeddings[0].length : 0);
  if (!dim) return null;

  const avg = new Array(dim).fill(0);
  indices.forEach((idx) => {
    const emb = embeddings[idx];
    if (!emb) return;
    for (let d = 0; d < dim; d++) {
      avg[d] += emb[d];
    }
  });

  for (let d = 0; d < dim; d++) {
    avg[d] /= indices.length;
  }
  return avg;
}

// --- Генерация коммента «от ЛЛМ» (правилами) --- //
function generateLLMComment(useCase, queryText, wine, similarity) {
  const simPercent = (similarity * 100).toFixed(1);
  const price = wine.price ? `$${wine.price}` : "не указана";
  const variety = wine.variety || "не указан";
  const country = wine.country || "не указана";
  const title = wine.title || "это вино";

  let prefix = "";

  if (useCase === 1) {
    prefix = `Это вино семантически близко к твоему описанию (${simPercent}% сходства). `;
  } else if (useCase === 2) {
    prefix = `Вино удовлетворяет заданным фильтрам и похоже на твой запрос (${simPercent}% сходства). `;
  } else if (useCase === 3) {
    prefix = `Это вино похоже на отмеченные тобой любимые (${simPercent}% сходства). `;
  }

  const tail =
    `Сорт: ${variety}. Страна: ${country}. Ориентировочная цена: ${price}. ` +
    `Информация из датафрейма (описание, регион, винодельня) использовалась при построении эмбеддингов.`;

  return prefix + tail;
}

// --- Рендер результатов --- //
function renderResults(container, items, useCase, queryText) {
  container.innerHTML = "";

  if (!items.length) {
    const p = document.createElement("p");
    p.textContent = "Ничего не найдено по заданным условиям.";
    container.appendChild(p);
    return;
  }

  items.forEach(({ wine, similarity }) => {
    const div = document.createElement("div");
    div.className = "result-item";

    const titleEl = document.createElement("div");
    titleEl.className = "result-title";
    titleEl.textContent = wine.title || "Без названия";

    const metaEl = document.createElement("div");
    metaEl.className = "result-meta";

    const metaParts = [];
    if (wine.winery) metaParts.push(String(wine.winery));
    if (wine.variety) metaParts.push(String(wine.variety));
    if (wine.country) metaParts.push(String(wine.country));
    if (wine.region_1) metaParts.push(String(wine.region_1));
    if (wine.price != null && !isNaN(wine.price)) metaParts.push(`$${wine.price}`);

    metaEl.textContent = metaParts.join(" • ");

    const descEl = document.createElement("div");
    descEl.className = "result-description";
    if (wine.description) {
      descEl.textContent = String(wine.description);
    }

    const simEl = document.createElement("div");
    simEl.className = "result-similarity";
    simEl.textContent = `Косинусное сходство: ${(similarity * 100).toFixed(1)}%`;

    const commentEl = document.createElement("div");
    commentEl.className = "result-comment";
    commentEl.textContent = generateLLMComment(useCase, queryText, wine, similarity);

    div.appendChild(titleEl);
    div.appendChild(metaEl);
    if (wine.description) div.appendChild(descEl);
    div.appendChild(simEl);
    div.appendChild(commentEl);

    container.appendChild(div);
  });
}

/* ---------- ИНИЦИАЛИЗАЦИЯ UI ---------- */

// Вкладки
function initTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  const contents = document.querySelectorAll(".tab-content");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;

      buttons.forEach((b) => b.classList.remove("active"));
      contents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(tab).classList.add("active");
    });
  });
}

// Фильтры из df.csv (никаких демо-данных)
function initFilters() {
  const countrySelect = document.getElementById("filter-country");
  const varietySelect = document.getElementById("filter-variety");

  const countries = new Set();
  const varieties = new Set();

  wines.forEach((w) => {
    if (w.country) countries.add(String(w.country).trim());
    if (w.variety) varieties.add(String(w.variety).trim());
  });

  Array.from(countries)
    .sort((a, b) => a.localeCompare(b))
    .forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      countrySelect.appendChild(opt);
    });

  Array.from(varieties)
    .sort((a, b) => a.localeCompare(b))
    .forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      varietySelect.appendChild(opt);
    });
}

// Список вин для выбора любимых (кейс 3)
function initFavoritesList() {
  const container = document.getElementById("favorites-list");
  container.innerHTML = "";

  // Показываем все вина из датафрейма (можно ограничить, если их очень много)
  wines.forEach((w) => {
    const item = document.createElement("div");
    item.className = "favorites-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = w._index;

    const label = document.createElement("label");
    const name = w.title || "Без названия";

    const metaParts = [];
    if (w.country) metaParts.push(w.country);
    if (w.variety) metaParts.push(w.variety);
    if (w.price != null && !isNaN(w.price)) metaParts.push(`$${w.price}`);

    label.textContent = `${name} (${metaParts.join(" • ")})`;

    item.appendChild(checkbox);
    item.appendChild(label);
    container.appendChild(item);
  });
}

// Обработчики трёх юзкейсов
function initUseCases() {
  // --- Кейc 1: по описанию --- //
  document.getElementById("case1-run").addEventListener("click", async () => {
    const queryEl = document.getElementById("case1-query");
    const resultsEl = document.getElementById("case1-results");
    const button = document.getElementById("case1-run");

    const query = queryEl.value.trim();
    if (!query) {
      resultsEl.textContent = "Пожалуйста, введи описание вина.";
      return;
    }

    try {
      button.disabled = true;
      button.textContent = "Считаем эмбеддинг и ищем...";

      const queryEmbedding = await getQueryEmbedding(query);
      const items = getTopKSimilar(queryEmbedding);
      renderResults(resultsEl, items, 1, query);
    } catch (err) {
      console.error(err);
      resultsEl.textContent = "Ошибка при поиске: " + err.message;
    } finally {
      button.disabled = false;
      button.textContent = "Найти похожие вина";
    }
  });

  // --- Кейc 2: по описанию + фильтры --- //
  document.getElementById("case2-run").addEventListener("click", async () => {
    const queryEl = document.getElementById("case2-query");
    const resultsEl = document.getElementById("case2-results");
    const button = document.getElementById("case2-run");

    const query = queryEl.value.trim();
    const country = document.getElementById("filter-country").value;
    const variety = document.getElementById("filter-variety").value;
    const maxPriceStr = document.getElementById("filter-max-price").value;
    const maxPrice = maxPriceStr ? Number(maxPriceStr) : null;

    if (!query) {
      resultsEl.textContent = "Пожалуйста, введи описание вина (query) для гибридного поиска.";
      return;
    }

    try {
      button.disabled = true;
      button.textContent = "Считаем эмбеддинг и ищем...";

      const queryEmbedding = await getQueryEmbedding(query);

      const items = getTopKSimilar(queryEmbedding, window.DEFAULT_TOP_K || 10, {
        filterCountry: country,
        filterVariety: variety,
        maxPrice: maxPrice,
      });

      renderResults(resultsEl, items, 2, query);
    } catch (err) {
      console.error(err);
      resultsEl.textContent = "Ошибка при поиске: " + err.message;
    } finally {
      button.disabled = false;
      button.textContent = "Найти вина с фильтрами";
    }
  });

  // --- Кейc 3: похожие на выбранные любимые --- //
  document.getElementById("case3-run").addEventListener("click", async () => {
    const resultsEl = document.getElementById("case3-results");
    const button = document.getElementById("case3-run");
    const container = document.getElementById("favorites-list");

    const checked = Array.from(container.querySelectorAll("input[type='checkbox']:checked"));
    if (!checked.length) {
      resultsEl.textContent = "Отметь хотя бы одно вино из списка.";
      return;
    }

    const indices = checked.map((cb) => Number(cb.value));

    try {
      button.disabled = true;
      button.textContent = "Считаем средний эмбеддинг и ищем...";

      const avgEmbedding = averageEmbeddings(indices);
      if (!avgEmbedding) {
        resultsEl.textContent = "Не удалось создать эмбеддинг для выбранных вин.";
        return;
      }

      const exclude = new Set(indices);
      const items = getTopKSimilar(avgEmbedding, window.DEFAULT_TOP_K || 10, {
        excludeIndices: exclude,
      });

      renderResults(resultsEl, items, 3, "(похожие на любимые)");
    } catch (err) {
      console.error(err);
      resultsEl.textContent = "Ошибка при поиске: " + err.message;
    } finally {
      button.disabled = false;
      button.textContent = "Найти похожие на выбранные";
    }
  });
}

// Запуск после загрузки DOM
document.addEventListener("DOMContentLoaded", () => {
  loadData(); // Грузим df.csv и wine_embeddings.json и инициализируем интерфейс
});
