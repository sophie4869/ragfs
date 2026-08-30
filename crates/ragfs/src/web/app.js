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

function rawBrowserUrl(file) {
  const url = rawUrl(file.relative_path);
  return isMarkdown(file) ? `${url}?raw=1` : url;
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

function fileExtension(path) {
  const name = (path || "").split("/").pop() || "";
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot + 1).toLowerCase();
}

function fileText(file) {
  return file.text || file.chunks.map((chunk) => chunk.content).join("\n\n");
}

function isMarkdown(file) {
  const mime = (file.mime_type || "").toLowerCase();
  const ext = fileExtension(file.relative_path || file.file);
  return mime.includes("markdown") || ext === "md" || ext === "markdown";
}

function isPdf(file) {
  const mime = (file.mime_type || "").toLowerCase();
  return mime === "application/pdf" || fileExtension(file.relative_path || file.file) === "pdf";
}

function appendInline(parent, text) {
  const pattern = /(!?\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) {
      parent.append(document.createTextNode(text.slice(last, match.index)));
    }
    parent.append(inlineNode(match[0]));
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    parent.append(document.createTextNode(text.slice(last)));
  }
}

function inlineNode(markup) {
  if (markup.startsWith("`") && markup.endsWith("`")) {
    const code = document.createElement("code");
    code.textContent = markup.slice(1, -1);
    return code;
  }

  if (markup.startsWith("**") && markup.endsWith("**")) {
    const strong = document.createElement("strong");
    appendInline(strong, markup.slice(2, -2));
    return strong;
  }

  if (markup.startsWith("*") && markup.endsWith("*")) {
    const em = document.createElement("em");
    appendInline(em, markup.slice(1, -1));
    return em;
  }

  const markdownLink = markup.match(/^!?\[([^\]]+)\]\(([^)]+)\)$/);
  if (markdownLink) {
    const [, label, href] = markdownLink;
    if (markup.startsWith("!")) {
      const img = document.createElement("img");
      img.alt = label;
      img.src = href;
      return img;
    }
    const link = document.createElement("a");
    link.textContent = label;
    link.href = href;
    link.target = "_blank";
    link.rel = "noreferrer";
    return link;
  }

  if (markup.startsWith("[[")) {
    const span = document.createElement("span");
    span.className = "wikilink";
    span.textContent = markup.slice(2, -2).split("|").pop();
    return span;
  }

  return document.createTextNode(markup);
}

function renderText(text) {
  const pre = document.createElement("pre");
  pre.textContent = text || "";
  preview.append(pre);
}

function renderMarkdown(text) {
  const article = document.createElement("article");
  article.className = "markdown-body";
  const lines = (text || "").replace(/\r\n?/g, "\n").split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }

    if (line.trimStart().startsWith("```")) {
      const language = line.trim().slice(3).trim();
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trimStart().startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) {
        i += 1;
      }
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      if (language) {
        code.dataset.language = language;
      }
      code.textContent = codeLines.join("\n");
      pre.append(code);
      article.append(pre);
      continue;
    }

    if (isTableStart(lines, i)) {
      const tableLines = [lines[i], lines[i + 1]];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        tableLines.push(lines[i]);
        i += 1;
      }
      article.append(renderTable(tableLines));
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length, 6);
      const node = document.createElement(`h${level}`);
      appendInline(node, heading[2].trim());
      article.append(node);
      i += 1;
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quote = document.createElement("blockquote");
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        const p = document.createElement("p");
        appendInline(p, lines[i].replace(/^\s*>\s?/, ""));
        quote.append(p);
        i += 1;
      }
      article.append(quote);
      continue;
    }

    if (/^\s*([-*+])\s+/.test(line) || /^\s*\d+\.\s+/.test(line)) {
      const ordered = /^\s*\d+\.\s+/.test(line);
      const list = document.createElement(ordered ? "ol" : "ul");
      const itemPattern = ordered ? /^\s*\d+\.\s+/ : /^\s*[-*+]\s+/;
      while (i < lines.length && itemPattern.test(lines[i])) {
        const item = document.createElement("li");
        appendInline(item, lines[i].replace(itemPattern, ""));
        list.append(item);
        i += 1;
      }
      article.append(list);
      continue;
    }

    const paragraphLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !isBlockStart(lines, i)
    ) {
      paragraphLines.push(lines[i].trim());
      i += 1;
    }
    const paragraph = document.createElement("p");
    appendInline(paragraph, paragraphLines.join(" "));
    article.append(paragraph);
  }

  preview.append(article);
}

function isBlockStart(lines, index) {
  const line = lines[index] || "";
  return (
    line.trimStart().startsWith("```") ||
    /^(#{1,6})\s+/.test(line) ||
    /^\s*>\s?/.test(line) ||
    /^\s*([-*+])\s+/.test(line) ||
    /^\s*\d+\.\s+/.test(line) ||
    isTableStart(lines, index)
  );
}

function isTableStart(lines, index) {
  return (
    (lines[index] || "").includes("|") &&
    /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1] || "")
  );
}

function splitTableRow(line) {
  let cells = line.trim();
  if (cells.startsWith("|")) {
    cells = cells.slice(1);
  }
  if (cells.endsWith("|")) {
    cells = cells.slice(0, -1);
  }
  return cells.split("|").map((cell) => cell.trim());
}

function renderTable(lines) {
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  const headerRow = document.createElement("tr");

  splitTableRow(lines[0]).forEach((cell) => {
    const th = document.createElement("th");
    appendInline(th, cell);
    headerRow.append(th);
  });
  thead.append(headerRow);

  lines.slice(2).forEach((line) => {
    const row = document.createElement("tr");
    splitTableRow(line).forEach((cell) => {
      const td = document.createElement("td");
      appendInline(td, cell);
      row.append(td);
    });
    tbody.append(row);
  });

  table.append(thead, tbody);
  return table;
}

async function renderPdf(file) {
  const iframe = document.createElement("iframe");
  iframe.title = file.title;
  iframe.src = await fetchRawObjectUrl(file.relative_path);
  preview.append(iframe);
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

  if (isPdf(file)) {
    await renderPdf(file);
    return;
  }

  if (mime.startsWith("video/")) {
    const video = document.createElement("video");
    video.controls = true;
    video.src = await fetchRawObjectUrl(path);
    preview.append(video);
    return;
  }

  if (isMarkdown(file)) {
    renderMarkdown(fileText(file));
    return;
  }

  renderText(fileText(file));
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
    rawLink.href = rawBrowserUrl(file);
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

const initialOpenPath = new URLSearchParams(window.location.search).get("open");
if (initialOpenPath) {
  openFile(initialOpenPath);
} else {
  input.focus();
}
