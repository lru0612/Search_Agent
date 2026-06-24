"use strict";

const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const emptyEl = document.getElementById("empty");
const hovercard = document.getElementById("hovercard");

let sessionId = null;
let running = false;
// temporary custom models, in memory only (lost on refresh)
const customModels = [];
let defaultModelName = "Default";
let awaitingClarify = false; // next user input is a reply to ask_user
let abortCtrl = null;

// rendering state for the current turn
let cur = null; // {timelineEl, stepsEl, mdEl, answerText, citations, settled}
const assistantBlocks = new WeakMap();

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
function scrollBottom() { chatEl.scrollTop = chatEl.scrollHeight; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function safeHttpUrl(url) {
  const raw = String(url || "").trim();
  if (!/^https?:\/\//i.test(raw) || /[\u0000-\u001F\u007F]/.test(raw)) return null;
  try {
    const parsed = new URL(raw);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

function addUserMsg(text) {
  emptyEl?.classList.add("hidden");
  const m = el("div", "msg user");
  m.appendChild(el("div", "bubble", esc(text)));
  chatEl.appendChild(m);
  scrollBottom();
}

function newAssistantBlock() {
  const m = el("div", "msg assistant");
  const timeline = el("details", "timeline");
  timeline.open = true;
  timeline.innerHTML = `<summary><span class="chev">▶</span><span class="pulse"></span><span class="tl-title">Thinking…</span></summary><div class="steps"></div>`;
  const md = el("div", "md");
  m.appendChild(timeline);
  m.appendChild(md);
  chatEl.appendChild(m);
  cur = {
    timelineEl: timeline,
    stepsEl: timeline.querySelector(".steps"),
    titleEl: timeline.querySelector(".tl-title"),
    mdEl: md,
    msgEl: m,
    answerText: "",
    citations: [],
    settled: false, // got final answer / ask_user / error / cancelled
    t0: Date.now(),
  };
  assistantBlocks.set(m, cur);
  scrollBottom();
}

function addStep(tag, text, isAction) {
  if (!cur) return;
  const s = el("div", "step" + (isAction ? " action" : ""));
  s.appendChild(el("span", "tag", esc(tag)));
  s.appendChild(el("span", "txt", esc(text)));
  cur.stepsEl.appendChild(s);
  scrollBottom();
}

function finishTimeline(label) {
  if (!cur) return;
  cur.timelineEl.querySelector(".pulse")?.remove();
  const secs = ((Date.now() - cur.t0) / 1000).toFixed(0);
  cur.titleEl.textContent = `${label} (${secs}s)`;
  cur.timelineEl.open = false;
}

function showError(message) {
  if (!cur) return;
  cur.settled = true;
  finishTimeline("Failed");
  cur.mdEl.appendChild(el("div", "error-box", esc(message)));
  scrollBottom();
}

// ------------------------------------------------ markdown + citations

function renderAnswer() {
  if (!cur) return;
  const cleanHtml = DOMPurify.sanitize(marked.parse(cur.answerText));
  const tpl = document.createElement("template");
  tpl.innerHTML = cleanHtml;
  linkCitations(tpl.content, cur);
  cur.mdEl.replaceChildren(tpl.content);
  scrollBottom();
}

function renderSources() {
  if (!cur || !cur.citations.length) return;
  const box = el("div", "sources");
  box.appendChild(el("div", "label", `Sources · ${cur.citations.length}`));
  for (const c of cur.citations) {
    const url = safeHttpUrl(c.url);
    const item = el(url ? "a" : "div", "source-card");
    if (url) {
      item.href = url;
      item.target = "_blank";
      item.rel = "noopener noreferrer";
    }
    const title = el("div", "t");
    title.textContent = `[${c.id}] ${c.title || c.url}`;
    const urlText = el("div", "u");
    urlText.textContent = c.url || "";
    item.appendChild(title);
    item.appendChild(urlText);
    box.appendChild(item);
  }
  cur.mdEl.appendChild(box);
}

function linkCitations(root, turn) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!/\[\d{1,3}\]/.test(node.nodeValue || "")) return NodeFilter.FILTER_REJECT;
      if (node.parentElement?.closest("a, code, pre")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const textNodes = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) textNodes.push(node);

  for (const node of textNodes) {
    const frag = document.createDocumentFragment();
    const text = node.nodeValue || "";
    let last = 0;
    for (const match of text.matchAll(/\[(\d{1,3})\]/g)) {
      if (match.index > last) frag.appendChild(document.createTextNode(text.slice(last, match.index)));
      frag.appendChild(makeCitationNode(turn, match[1], match[0]));
      last = match.index + match[0].length;
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    node.replaceWith(frag);
  }
}

function makeCitationNode(turn, id, fallbackText) {
  const src = turn.citations.find(c => String(c.id) === String(id));
  const url = src ? safeHttpUrl(src.url) : null;
  if (!src || !url) return document.createTextNode(fallbackText);

  const a = document.createElement("a");
  a.className = "cite";
  a.dataset.id = String(id);
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.textContent = String(id);
  return a;
}

// citation hover card
document.addEventListener("mouseover", e => {
  const cite = e.target.closest?.(".cite");
  if (!cite) { hovercard.classList.add("hidden"); return; }
  const msg = cite.closest(".msg.assistant");
  const turn = msg ? assistantBlocks.get(msg) : null;
  const src = turn?.citations.find(c => String(c.id) === cite.dataset.id);
  if (!src) { hovercard.classList.add("hidden"); return; }

  const title = el("div", "t");
  title.textContent = src.title || src.url;
  const snippet = el("div", "s");
  snippet.textContent = src.snippet || "";
  hovercard.replaceChildren(title, snippet);
  hovercard.classList.remove("hidden");
  const r = cite.getBoundingClientRect();
  const left = Math.max(12, Math.min(r.left, innerWidth - hovercard.offsetWidth - 12));
  const top = r.top > hovercard.offsetHeight + 16 ? r.top - hovercard.offsetHeight - 8 : r.bottom + 8;
  hovercard.style.left = left + "px";
  hovercard.style.top = Math.max(12, Math.min(top, innerHeight - hovercard.offsetHeight - 12)) + "px";
});

// ------------------------------------------------ clarification card

function showClarify(data) {
  cur.settled = true;
  finishTimeline("Waiting for your input");
  const card = el("div", "clarify-card");
  card.appendChild(el("div", "q", esc(data.question)));
  if (data.options?.length) {
    const opts = el("div", "opts");
    for (const o of data.options) {
      const b = el("button", "opt-btn", esc(o));
      b.onclick = () => { inputEl.value = o; send(); };
      opts.appendChild(b);
    }
    card.appendChild(opts);
  }
  // inline reply box inside the card
  const row = el("div", "clarify-input");
  const field = el("input");
  field.type = "text";
  field.placeholder = "Type your answer…";
  const go = el("button", "clarify-send", `<svg viewBox="0 0 24 24" width="15" height="15"><path fill="currentColor" d="M3.4 20.4l17.4-7.5c.8-.4.8-1.5 0-1.8L3.4 3.6c-.7-.3-1.4.3-1.4 1l0 4.6c0 .5.4.9.9 1L15 12 2.9 13.8c-.5.1-.9.5-.9 1l0 4.6c0 .7.7 1.3 1.4 1z"/></svg>`);
  go.title = "Reply";
  const submit = () => {
    const v = field.value.trim();
    if (!v) return;
    inputEl.value = v;
    send();
  };
  go.onclick = submit;
  field.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); submit(); }
  });
  row.appendChild(field);
  row.appendChild(go);
  card.appendChild(row);
  chatEl.appendChild(card);
  field.focus();
  awaitingClarify = true;
  setRunning(false);
  scrollBottom();
}

// ------------------------------------------------ SSE handling

const ACTION_LABELS = {
  WebSearch: "Search",
  web_search: "Search",
  VisitPage: "Read",
  visit_page: "Read",
  AskUser: "Ask",
  ask_user: "Ask",
  CurateEvidence: "Curate",
  curate_evidence: "Curate",
  PruneCandidates: "Prune",
  prune_candidates: "Prune",
  VerifyClaim: "Verify",
  verify_claim: "Verify",
  Finish: "Finish",
  finish: "Finish",
};

function handleEvent(type, data) {
  switch (type) {
    case "session":
      sessionId = data.session_id;
      break;
    case "node":
      if (data.node === "clarify") addStep("Check", "Checking whether the query needs clarification");
      break;
    case "rewrite":
      addStep("Rewrite", (data.queries || []).join("  |  "));
      break;
    case "action": {
      const label = ACTION_LABELS[data.tool] || data.tool;
      let txt = "";
      if (data.tool === "WebSearch" || data.tool === "web_search") txt = data.args.query;
      else if (data.tool === "VisitPage" || data.tool === "visit_page") txt = `${data.args.url}${data.args.reason ? ` (${data.args.reason})` : ""}`;
      else if (data.tool === "CurateEvidence" || data.tool === "curate_evidence") txt = `C${data.args.candidate_id} · ${data.args.claim || ""}`;
      else if (data.tool === "PruneCandidates" || data.tool === "prune_candidates") txt = `${(data.args.candidate_ids || []).map(id => `C${id}`).join(", ")} · ${data.args.reason || ""}`;
      else if (data.tool === "VerifyClaim" || data.tool === "verify_claim") txt = `${data.args.claim || ""} · sources ${(data.args.source_ids || []).join(", ")}`;
      else if (data.tool === "Finish" || data.tool === "finish") txt = "Evidence is sufficient, preparing the answer";
      else txt = JSON.stringify(data.args);
      addStep(label, `Step ${data.step ?? "?"} · ${txt}`, true);
      break;
    }
    case "observation":
      addStep("Result", data.preview);
      break;
    case "candidate": {
      const latest = (data.latest || []).map(c => `C${c.id}: ${c.title || c.url}`).join("  |  ");
      addStep("Candidates", `${data.count || 0} active${latest ? ` · ${latest}` : ""}`);
      break;
    }
    case "curate": {
      const e = data.evidence || {};
      addStep("Curated", `E${e.id || "?"} · C${e.candidate_id || "?"} -> [${e.source_id || "?"}] · ${e.claim || ""}`, true);
      break;
    }
    case "prune": {
      const r = data.record || {};
      addStep("Pruned", `${(r.candidate_ids || []).map(id => `C${id}`).join(", ")} · ${r.reason || ""}`, true);
      break;
    }
    case "verify": {
      const r = data.record || {};
      addStep("Verified", `${r.verdict || "unknown"} · ${r.claim || ""}`, true);
      break;
    }
    case "reflect":
      addStep("Reflect", data.passed ? "Self-check passed" : `Gaps found: ${data.feedback || ""}`, true);
      break;
    case "budget_exhausted":
      addStep("Budget", "Search budget exhausted, forcing the answer phase", true);
      break;
    case "ask_user":
      showClarify(data);
      break;
    case "answer_chunk":
      if (cur && !cur.answerText) finishTimeline("Search complete");
      cur.answerText += data.text;
      renderAnswer();
      break;
    case "final_answer":
      cur.settled = true;
      cur.answerText = data.answer;
      cur.citations = data.citations || [];
      finishTimeline("Search complete");
      renderAnswer();
      renderSources();
      break;
    case "cancelled":
      cur.settled = true;
      finishTimeline("Stopped");
      break;
    case "error":
      showError(data.message || "Unknown error");
      break;
    case "done":
      setRunning(false);
      break;
  }
}

async function streamChat(body) {
  abortCtrl = new AbortController();
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: abortCtrl.signal,
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`Server returned HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse-starlette may use \r\n line endings; normalize before parsing
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const blocks = buf.split("\n\n");
    buf = blocks.pop();
    for (const block of blocks) {
      let ev = "message", dataStr = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      if (!dataStr) continue;
      try { handleEvent(ev, JSON.parse(dataStr)); } catch (e) { console.error(e, dataStr); }
    }
  }
}

function setRunning(v) {
  running = v;
  sendBtn.classList.toggle("hidden", v);
  stopBtn.classList.toggle("hidden", !v);
  inputEl.disabled = false;
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || running) return;
  inputEl.value = "";
  autoGrow();
  addUserMsg(text);
  setRunning(true);

  const isResume = awaitingClarify;
  awaitingClarify = false;
  newAssistantBlock();
  try {
    await streamChat({ message: text, session_id: sessionId, resume: isResume, model_override: currentModelOverride() });
    // stream ended without any terminal event -> surface it instead of spinning forever
    if (cur && !cur.settled) {
      showError("The stream ended unexpectedly without a result. Check the server logs.");
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      showError("Connection failed: " + e.message);
    }
  } finally {
    if (!awaitingClarify) setRunning(false);
  }
}

stopBtn.onclick = async () => {
  abortCtrl?.abort();
  if (sessionId) fetch(`/api/cancel/${sessionId}`, { method: "POST" });
  if (cur && !cur.settled) { cur.settled = true; finishTimeline("Stopped"); }
  setRunning(false);
};

document.getElementById("newChatBtn").onclick = () => location.reload();

// ------------------------------------------------ model selector

const modelSelect = document.getElementById("modelSelect");
const modelModal = document.getElementById("modelModal");

function refreshModelSelect() {
  const sel = modelSelect.value;
  modelSelect.innerHTML = "";
  const def = el("option", "", esc(`${defaultModelName} (default)`));
  def.value = "";
  modelSelect.appendChild(def);
  customModels.forEach((m, i) => {
    const o = el("option", "", esc(m.model));
    o.value = String(i);
    modelSelect.appendChild(o);
  });
  if ([...modelSelect.options].some(o => o.value === sel)) modelSelect.value = sel;
}

function currentModelOverride() {
  const v = modelSelect.value;
  if (v === "") return null;
  const m = customModels[Number(v)];
  return m ? { model: m.model, api_key: m.api_key, base_url: m.base_url } : null;
}

fetch("/api/config").then(r => r.json()).then(d => {
  defaultModelName = d.default_model || "Default";
  refreshModelSelect();
}).catch(() => refreshModelSelect());

document.getElementById("addModelBtn").onclick = () => {
  modelModal.classList.remove("hidden");
  document.getElementById("mName").focus();
};
document.getElementById("mCancel").onclick = () => modelModal.classList.add("hidden");
modelModal.addEventListener("click", e => { if (e.target === modelModal) modelModal.classList.add("hidden"); });
document.getElementById("mSave").onclick = () => {
  const model = document.getElementById("mName").value.trim();
  const base_url = document.getElementById("mBase").value.trim();
  const api_key = document.getElementById("mKey").value.trim();
  if (!model) { document.getElementById("mName").focus(); return; }
  customModels.push({ model, base_url, api_key });
  refreshModelSelect();
  modelSelect.value = String(customModels.length - 1);
  modelModal.classList.add("hidden");
  for (const id of ["mName", "mBase", "mKey"]) document.getElementById(id).value = "";
};

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
}
inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
sendBtn.onclick = send;
