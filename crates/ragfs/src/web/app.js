const state = {
  token: localStorage.getItem("ragfsToken") || "",
  activePath: "",
  activeObjectUrl: "",
};

const form = document.querySelector("#searchForm");
const input = document.querySelector("#queryInput");
const results = document.querySelector("#results");
const statusLine = document.querySelector("#statusLine");
const tokenButton = document.querySelector("#tokenButton");
const emptyState = document.querySelector("#emptyState");
const reader = document.querySelector("#reader");
const readerTitle = document.querySelector("#readerTitle");
const readerPath = document.querySelector("#readerPath");
const rawLink = document.querySelector("#rawLink");
const preview = document.querySelector("#preview");

function apiHeaders() {
  const headers = {};
  if (state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }
  return headers;
}

async function requestJson(url) {
  const response = await fetch(url, { headers: apiHeaders() });
  if (response.status === 401) {
    askForToken();
    throw new Error("Unauthorized");
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function askForToken() {
  const token = window.prompt("API token", state.token);
  if (token === null) {
    return;
  }
  state.token = token.trim();
  if (state.token) {
    localStorage.setItem("ragfsToken", state.token);
  } else {
    localStorage.removeItem("ragfsToken");
  }
}

function showNotice(message) {
  results.innerHTML = "";
  const item = document.createElement("li");
  item.className = "notice";
  item.textContent = message;
  results.append(item);
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function displayStatus(data) {
  statusLine.textContent = `${data.total_files} files / ${data.total_chunks} chunks`;
}

async function loadStatus() {
  try {
    const data = await requestJson("/api/status");
    displayStatus(data);
  } catch (error) {
    statusLine.textContent = error.message;
  }
}

function resultButton(item) {
  const li = document.createElement("li");
  li.className = "result";
  li.dataset.path = item.relative_path || "";

  const button = document.createElement("button");
  button.type = "button";

  const row = document.createElement("div");
  row.className = "result-title-row";

  const title = document.createElement("div");
  title.className = "result-title";
  title.textContent = item.title || item.relative_path || item.file;

  const score = document.createElement("div");
  score.className = "score";
  score.textContent = Number(item.score).toFixed(2);

  const path = document.createElement("div");
  path.className = "result-path";
  path.textContent = item.relative_path || item.file;

  const snippet = document.createElement("div");
  snippet.className = "snippet";
  snippet.textContent = item.content || "";

  const reason = document.createElement("div");
  reason.className = "reason";
  reason.textContent = item.reason || "semantic";

  row.append(title, score);
  button.append(row, path, reason, snippet);
  button.addEventListener("click", () => openFile(item.relative_path));
  li.append(button);
  return li;
}

async function search(query) {
  results.innerHTML = "";
  if (!query.trim()) {
    return;
  }

  statusLine.textContent = "Searching...";
  try {
    const url = `/api/search?q=${encodeURIComponent(query)}&limit=25`;
    const data = await requestJson(url);
    if (data.results.length === 0) {
      showNotice("No results.");
    } else {
      results.append(...data.results.map(resultButton));
    }
    statusLine.textContent = `${data.results.length} results`;
  } catch (error) {
    showNotice(error.message);
    statusLine.textContent = "Search failed";
  }
}

function setActive(path) {
  state.activePath = path;
  document.querySelectorAll(".result").forEach((node) => {
    node.classList.toggle("is-active", node.dataset.path === path);
  });
}

function clearPreview() {
  if (state.activeObjectUrl) {
    URL.revokeObjectURL(state.activeObjectUrl);
    state.activeObjectUrl = "";
  }
  preview.innerHTML = "";
}

function rawUrl(path) {
  return `/raw/${path.split("/").map(encodeURIComponent).join("/")}`;
}

async function fetchRawObjectUrl(path) {
  const response = await fetch(rawUrl(path), { headers: apiHeaders() });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const blob = await response.blob();
  state.activeObjectUrl = URL.createObjectURL(blob);
  return state.activeObjectUrl;
}

function renderText(text) {
  const pre = document.createElement("pre");
  pre.textContent = text || "";
  preview.append(pre);
}

async function renderPreview(file) {
  clearPreview();
  const path = file.relative_path;
  const mime = file.mime_type || "";

  if (mime.startsWith("image/")) {
    const img = document.createElement("img");
    img.alt = file.title;
    img.src = await fetchRawObjectUrl(path);
    preview.append(img);
    return;
  }

  if (mime === "application/pdf") {
    const iframe = document.createElement("iframe");
    iframe.title = file.title;
    iframe.src = await fetchRawObjectUrl(path);
    preview.append(iframe);
    return;
  }

  if (mime.startsWith("video/")) {
    const video = document.createElement("video");
    video.controls = true;
    video.src = await fetchRawObjectUrl(path);
    preview.append(video);
    return;
  }

  renderText(file.text || file.chunks.map((chunk) => chunk.content).join("\n\n"));
}

async function openFile(path) {
  if (!path) {
    return;
  }

  setActive(path);
  emptyState.hidden = true;
  reader.hidden = false;
  readerTitle.textContent = "Loading...";
  readerPath.textContent = path;
  clearPreview();

  try {
    const file = await requestJson(`/api/files/${path.split("/").map(encodeURIComponent).join("/")}`);
    readerTitle.textContent = file.title;
    readerPath.textContent = `${file.relative_path} · ${formatBytes(file.size_bytes)}`;
    rawLink.href = rawUrl(file.relative_path);
    await renderPreview(file);
  } catch (error) {
    readerTitle.textContent = "Unable to open";
    renderText(error.message);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  search(input.value);
});

tokenButton.addEventListener("click", () => {
  askForToken();
  loadStatus();
});

rawLink.addEventListener("click", async (event) => {
  if (!state.token || !state.activePath) {
    return;
  }
  event.preventDefault();
  try {
    const url = state.activeObjectUrl || (await fetchRawObjectUrl(state.activePath));
    window.open(url, "_blank", "noopener");
  } catch (error) {
    renderText(error.message);
  }
});

loadStatus();
input.focus();
