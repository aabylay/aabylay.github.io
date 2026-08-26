/* GLS correlation demo — logic matches gls/GLS_DEMO_SPEC.md */

const BLOB_C = { x: 0.35, y: 0.65 };
const BLOB_S = 0.18;
const RHO_NEUTRAL = 0.05;
const HIT_PX = 32;
const MODE_HINTS = {
  independent:
    "Filter scores ignore (x, y). Expect ρ_q ≈ 0 anywhere; noise grows as σ_g or k shrinks.",
  clustered:
    "Valid points concentrate near the blob. Query inside → enrichment (easier); outside → depletion (harder).",
  gradient:
    "Valid points prefer larger x. ρ_q changes as the query moves left or right.",
};

const DEFAULTS = {
  n: 400,
  sigma: 0.2,
  mode: "clustered",
  qx: BLOB_C.x,
  qy: BLOB_C.y,
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pedagogicalK(n) {
  return Math.max(1, Math.round(0.1 * n));
}

function snapN(n) {
  n = Math.round(Number(n) / 50) * 50;
  if (!Number.isFinite(n)) n = DEFAULTS.n;
  return Math.min(1000, Math.max(100, n));
}

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.min(1, Math.max(0, x));
}

function snapSigma(s) {
  s = Math.round(Number(s) * 100) / 100;
  if (!Number.isFinite(s)) s = DEFAULTS.sigma;
  return clamp01(s);
}

function randomSeed() {
  if (window.crypto && crypto.getRandomValues) {
    const buf = new Uint32Array(1);
    crypto.getRandomValues(buf);
    return buf[0] >>> 0;
  }
  return (Math.random() * 0xffffffff) >>> 0;
}

function defaultAseed(seed) {
  return (seed + 1) >>> 0;
}

/** Stable ρ; do not form r first when sigma_l === 0. */
function rhoFromSelectivities(sigmaL, sigmaG) {
  if (!(sigmaG > 0)) return null; // empty filter: r and rho are n/a
  return (sigmaL - sigmaG) / (sigmaL + sigmaG);
}

function generatePoints(n, seed) {
  const rng = mulberry32(seed);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = rng();
    y[i] = rng();
  }
  return { x, y };
}

function generateScores(mode, x, y, aseed) {
  const n = x.length;
  const a = new Float64Array(n);
  if (mode === "gradient") {
    for (let i = 0; i < n; i++) a[i] = x[i];
    return a;
  }
  if (mode === "clustered") {
    const inv = 1 / (2 * BLOB_S * BLOB_S);
    for (let i = 0; i < n; i++) {
      const dx = x[i] - BLOB_C.x;
      const dy = y[i] - BLOB_C.y;
      a[i] = Math.exp(-(dx * dx + dy * dy) * inv);
    }
    return a;
  }
  const rng = mulberry32(aseed);
  for (let i = 0; i < n; i++) a[i] = rng();
  return a;
}

function applyFilter(a, nG) {
  const n = a.length;
  const pass = new Uint8Array(n);
  nG = Math.min(n, Math.max(0, nG));
  if (nG <= 0) return pass;
  if (nG >= n) {
    pass.fill(1);
    return pass;
  }
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((i, j) => {
    const d = a[j] - a[i];
    if (d !== 0) return d < 0 ? -1 : 1;
    return i - j;
  });
  for (let t = 0; t < nG; t++) pass[idx[t]] = 1;
  return pass;
}

function knn(qx, qy, x, y, k) {
  const n = x.length;
  k = Math.min(n, Math.max(1, k));
  const dist = new Float64Array(n);
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const dx = x[i] - qx;
    const dy = y[i] - qy;
    dist[i] = dx * dx + dy * dy;
    idx[i] = i;
  }
  idx.sort((i, j) => {
    const d = dist[i] - dist[j];
    if (d !== 0) return d < 0 ? -1 : 1;
    return i - j;
  });
  const neighbors = idx.subarray(0, k);
  const radius = Math.sqrt(dist[neighbors[k - 1]]);
  return { neighbors, radius };
}

function fmt(x, d) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "n/a";
  return x.toFixed(d);
}

const state = {
  n: DEFAULTS.n,
  sigma: DEFAULTS.sigma,
  mode: DEFAULTS.mode,
  seed: 7,
  aseed: defaultAseed(7),
  qx: DEFAULTS.qx,
  qy: DEFAULTS.qy,
  k: pedagogicalK(DEFAULTS.n),
  kAuto: true,
  x: null,
  y: null,
  a: null,
};

const el = {};
let dragging = false;
let grabDX = 0;
let grabDY = 0;
let needsURL = false;
let raf = 0;
let plotLayout = { size: 0, pad: 16, plotSize: 0 };

function parseURL() {
  const q = new URLSearchParams(location.search);
  if (q.has("n")) state.n = snapN(q.get("n"));
  if (q.has("sigma")) state.sigma = snapSigma(q.get("sigma"));
  const mode = (q.get("mode") || "").toLowerCase();
  if (mode === "independent" || mode === "clustered" || mode === "gradient") {
    state.mode = mode;
  }
  if (q.has("seed")) {
    const s = Number(q.get("seed"));
    state.seed = Number.isFinite(s) ? s >>> 0 : randomSeed();
  } else {
    state.seed = randomSeed();
  }
  if (q.has("aseed")) {
    const s = Number(q.get("aseed"));
    state.aseed = Number.isFinite(s) ? s >>> 0 : defaultAseed(state.seed);
  } else {
    state.aseed = defaultAseed(state.seed);
  }
  if (q.has("qx")) state.qx = clamp01(Number(q.get("qx")));
  if (q.has("qy")) state.qy = clamp01(Number(q.get("qy")));
  if (q.has("k")) {
    const k = Math.round(Number(q.get("k")));
    if (Number.isFinite(k)) {
      state.k = Math.min(state.n, Math.max(1, k));
      state.kAuto = state.k === pedagogicalK(state.n);
    }
  } else {
    state.k = pedagogicalK(state.n);
    state.kAuto = true;
  }
}

function trimNum(x, d) {
  return String(Number(Number(x).toFixed(d)));
}

function writeURL() {
  const params = new URLSearchParams();
  params.set("n", String(state.n));
  params.set("sigma", trimNum(state.sigma, 2));
  params.set("mode", state.mode);
  params.set("seed", String(state.seed));
  params.set("qx", trimNum(state.qx, 4));
  params.set("qy", trimNum(state.qy, 4));
  const autoK = pedagogicalK(state.n);
  if (!state.kAuto && state.k !== autoK) params.set("k", String(state.k));
  if (state.mode === "independent" && state.aseed !== defaultAseed(state.seed)) {
    params.set("aseed", String(state.aseed));
  }
  const qs = "?" + params.toString();
  if (qs !== location.search) {
    history.replaceState(null, "", qs);
  }
}

function rebuildPoints() {
  const pts = generatePoints(state.n, state.seed);
  state.x = pts.x;
  state.y = pts.y;
  rebuildScores();
}

function rebuildScores() {
  state.a = generateScores(state.mode, state.x, state.y, state.aseed);
}

function compute() {
  const nG = Math.min(state.n, Math.max(0, Math.round(state.sigma * state.n)));
  const sigmaG = nG / state.n;
  const pass = applyFilter(state.a, nG);
  const { neighbors, radius } = knn(state.qx, state.qy, state.x, state.y, state.k);
  const inKnn = new Uint8Array(state.n);
  let nL = 0;
  for (let t = 0; t < neighbors.length; t++) {
    const i = neighbors[t];
    inKnn[i] = 1;
    if (pass[i]) nL += 1;
  }
  const sigmaL = nL / state.k;
  const rho = rhoFromSelectivities(sigmaL, sigmaG);
  const r = sigmaG > 0 ? sigmaL / sigmaG : null;
  return { nG, nL, sigmaG, sigmaL, r, rho, pass, inKnn, radius };
}

function toCanvas(x, y) {
  const { pad, plotSize } = plotLayout;
  return [pad + x * plotSize, pad + (1 - y) * plotSize];
}

function eventToPlot(e) {
  const rect = el.plot.getBoundingClientRect();
  const cssX = e.clientX - rect.left;
  const cssY = e.clientY - rect.top;
  const { pad, plotSize } = plotLayout;
  return {
    x: clamp01((cssX - pad) / plotSize),
    y: clamp01(1 - (cssY - pad) / plotSize),
    cssX,
    cssY,
  };
}

function drawStar(ctx, x, y, outer) {
  const spikes = 5;
  const inner = outer * 0.4;
  ctx.beginPath();
  for (let i = 0; i < spikes * 2; i++) {
    const angle = (i * Math.PI) / spikes - Math.PI / 2;
    const rad = i % 2 === 0 ? outer : inner;
    const px = x + Math.cos(angle) * rad;
    const py = y + Math.sin(angle) * rad;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = "#e05050";
  ctx.fill();
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawPoint(ctx, x, y, radius, fill, stroke, lineWidth) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function syncCanvasSize() {
  const canvas = el.plot;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 8) return false;
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const size = rect.width;
  const pad = Math.max(16, Math.min(24, size * 0.045));
  plotLayout = { size, pad, plotSize: size - 2 * pad, dpr };
  return true;
}

function draw(m) {
  if (!syncCanvasSize()) return;
  const canvas = el.plot;
  const ctx = canvas.getContext("2d");
  const { size, pad, plotSize, dpr } = plotLayout;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  ctx.fillStyle = "#fafafa";
  ctx.fillRect(pad, pad, plotSize, plotSize);

  const rPt = (state.n > 700 ? 2.35 : state.n > 400 ? 2.7 : 3.15) * 1.5;
  const [qx, qy] = toCanvas(state.qx, state.qy);

  ctx.save();
  ctx.beginPath();
  ctx.rect(pad, pad, plotSize, plotSize);
  ctx.clip();
  if (m.radius > 0 && Number.isFinite(m.radius)) {
    const rCss = m.radius * plotSize;
    ctx.beginPath();
    ctx.arc(qx, qy, rCss, 0, Math.PI * 2);
    ctx.strokeStyle = "#6a6ad4";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const { x, y, n } = state;
  const { pass, inKnn } = m;
  for (let i = 0; i < n; i++) {
    if (inKnn[i]) continue;
    const [px, py] = toCanvas(x[i], y[i]);
    if (pass[i]) drawPoint(ctx, px, py, rPt, "#7dcc7d", "#333", 1);
    else drawPoint(ctx, px, py, rPt, "#c8c8c8", "#333", 1);
  }
  const rKnn = rPt + 0.5;
  for (let i = 0; i < n; i++) {
    if (!inKnn[i]) continue;
    const [px, py] = toCanvas(x[i], y[i]);
    if (pass[i]) drawPoint(ctx, px, py, rKnn, "#7dcc7d", "#6a6ad4", 2);
    else drawPoint(ctx, px, py, rKnn, "#6a6ad4", "#222", 1.5);
  }
  ctx.restore();

  ctx.strokeStyle = "#002856";
  ctx.lineWidth = 1.25;
  ctx.strokeRect(pad + 0.5, pad + 0.5, plotSize - 1, plotSize - 1);

  ctx.fillStyle = "#5c6570";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("0", pad - 1, pad + plotSize + 12);
  ctx.fillText("1", pad + plotSize - 6, pad + plotSize + 12);
  ctx.fillText("1", pad - 10, pad + 4);
  ctx.fillText("0", pad - 10, pad + plotSize);

  const starR = Math.max(10, Math.min(14, size * 0.028));
  drawStar(ctx, qx, qy, starR);
}

function renderPanel(m) {
  const sg = fmt(m.sigmaG, 3);
  const sl = fmt(m.sigmaL, 3);
  el.termG.innerHTML =
    "σ<sub>g</sub> = N<sub>f</sub>/N = " + m.nG + "/" + state.n + " = " + sg;
  el.termL.innerHTML =
    "σ<sub>l</sub> = N<sub>l</sub>/k = " + m.nL + "/" + state.k + " = " + sl;

  if (!(m.sigmaG > 0)) {
    el.termR.innerHTML = "r = σ<sub>l</sub>/σ<sub>g</sub> = n/a";
    el.termRho.innerHTML = "ρ<sub>q</sub> = n/a";
  } else {
    el.termR.innerHTML =
      "r = σ<sub>l</sub>/σ<sub>g</sub> = " + sl + "/" + sg + " = " + fmt(m.r, 3);
    el.termRho.innerHTML = "ρ<sub>q</sub> = " + fmt(m.rho, 3);
  }

  if (m.rho === null) {
    el.badge.hidden = true;
    el.rhoKnob.classList.add("is-na");
  } else {
    el.badge.hidden = false;
    el.rhoKnob.classList.remove("is-na");
    const t = (m.rho + 1) / 2;
    el.rhoKnob.style.left = 100 * Math.min(1, Math.max(0, t)) + "%";
    if (m.rho > RHO_NEUTRAL) {
      el.badge.className = "badge enrich";
      el.badge.textContent = "enrichment / easier";
    } else if (m.rho < -RHO_NEUTRAL) {
      el.badge.className = "badge deplete";
      el.badge.textContent = "depletion / harder";
    } else {
      el.badge.className = "badge neutral";
      el.badge.textContent = "neutral / independent";
    }
  }

  const expectedLocal = m.sigmaG * state.k;
  if (m.sigmaG > 0 && expectedLocal <= 3) {
    el.finiteK.hidden = false;
    el.finiteK.textContent =
      "Finite-k artifact: expected local pass count σ_g k = " +
      fmt(expectedLocal, 2) +
      " ≲ 3. A neighborhood of size k often contains fewer valid points than σ_g k, so ρ_q drifts negative even when the filter is independent of geometry.";
  } else {
    el.finiteK.hidden = true;
  }

  el.modeHint.textContent = MODE_HINTS[state.mode] || "";
}

function syncControls() {
  el.sigma.value = String(state.sigma);
  el.sigmaVal.textContent = state.sigma.toFixed(2);
  el.mode.value = state.mode;
  el.n.value = String(state.n);
  el.nVal.textContent = String(state.n);
  el.k.max = String(state.n);
  el.k.value = String(state.k);
  const autoK = pedagogicalK(state.n);
  el.kVal.textContent = String(state.k);
  el.kNote.textContent =
    "default k = round(0.1 N) = " + autoK + (state.k === autoK ? "" : "  (slider overrides)");
}

function render() {
  raf = 0;
  const m = compute();
  draw(m);
  renderPanel(m);
}

function scheduleRender() {
  if (raf) return;
  raf = requestAnimationFrame(render);
}

function setQ(x, y) {
  state.qx = clamp01(x);
  state.qy = clamp01(y);
  scheduleRender();
  needsURL = true;
}

function onNChanged(n) {
  state.n = snapN(n);
  if (state.kAuto) state.k = pedagogicalK(state.n);
  else state.k = Math.min(state.n, Math.max(1, state.k));
  rebuildPoints();
  syncControls();
  render();
  writeURL();
}

function bind() {
  el.plot = document.getElementById("plot");
  el.termG = document.getElementById("term-g");
  el.termL = document.getElementById("term-l");
  el.termR = document.getElementById("term-r");
  el.termRho = document.getElementById("term-rho");
  el.badge = document.getElementById("badge");
  el.rhoKnob = document.getElementById("rho-knob");
  el.finiteK = document.getElementById("finite-k");
  el.modeHint = document.getElementById("mode-hint");
  el.sigma = document.getElementById("sigma");
  el.sigmaVal = document.getElementById("sigma-val");
  el.mode = document.getElementById("mode");
  el.n = document.getElementById("n");
  el.nVal = document.getElementById("n-val");
  el.k = document.getElementById("k");
  el.kVal = document.getElementById("k-val");
  el.kNote = document.getElementById("k-note");

  el.sigma.addEventListener("input", () => {
    state.sigma = snapSigma(el.sigma.value);
    el.sigmaVal.textContent = state.sigma.toFixed(2);
    scheduleRender();
    needsURL = true;
  });
  el.sigma.addEventListener("change", writeURL);

  el.mode.addEventListener("change", () => {
    state.mode = el.mode.value;
    rebuildScores();
    el.modeHint.textContent = MODE_HINTS[state.mode] || "";
    render();
    writeURL();
  });

  el.n.addEventListener("input", () => onNChanged(el.n.value));

  el.k.addEventListener("input", () => {
    state.k = Math.min(state.n, Math.max(1, Math.round(Number(el.k.value))));
    state.kAuto = state.k === pedagogicalK(state.n);
    el.kVal.textContent = String(state.k);
    const autoK = pedagogicalK(state.n);
    el.kNote.textContent =
      "default k = round(0.1 N) = " + autoK + (state.k === autoK ? "" : "  (slider overrides)");
    scheduleRender();
    needsURL = true;
  });
  el.k.addEventListener("change", writeURL);

  document.getElementById("btn-resample").addEventListener("click", () => {
    state.seed = randomSeed();
    state.aseed = defaultAseed(state.seed);
    rebuildPoints();
    render();
    writeURL();
  });

  document.getElementById("btn-reshuffle").addEventListener("click", () => {
    state.aseed = randomSeed();
    rebuildScores();
    render();
    writeURL();
  });

  document.getElementById("btn-reset").addEventListener("click", () => {
    state.n = DEFAULTS.n;
    state.sigma = DEFAULTS.sigma;
    state.mode = DEFAULTS.mode;
    state.qx = DEFAULTS.qx;
    state.qy = DEFAULTS.qy;
    state.seed = randomSeed();
    state.aseed = defaultAseed(state.seed);
    state.k = pedagogicalK(state.n);
    state.kAuto = true;
    rebuildPoints();
    syncControls();
    render();
    writeURL();
  });

  el.plot.addEventListener("contextmenu", (e) => e.preventDefault());

  el.plot.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    el.plot.setPointerCapture(e.pointerId);
    dragging = true;
    const p = eventToPlot(e);
    const [qx, qy] = toCanvas(state.qx, state.qy);
    const dx = p.cssX - qx;
    const dy = p.cssY - qy;
    if (dx * dx + dy * dy > HIT_PX * HIT_PX) {
      grabDX = 0;
      grabDY = 0;
      setQ(p.x, p.y);
    } else {
      grabDX = state.qx - p.x;
      grabDY = state.qy - p.y;
      needsURL = true;
    }
  });

  el.plot.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const p = eventToPlot(e);
    setQ(p.x + grabDX, p.y + grabDY);
  });

  function endDrag(e) {
    if (!dragging) return;
    dragging = false;
    try {
      el.plot.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* already released */
    }
    if (needsURL) {
      needsURL = false;
      writeURL();
    }
  }

  el.plot.addEventListener("pointerup", endDrag);
  el.plot.addEventListener("pointercancel", endDrag);

  window.addEventListener("resize", scheduleRender);
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(scheduleRender).observe(el.plot);
  }
}

function boot() {
  parseURL();
  bind();
  rebuildPoints();
  syncControls();
  render();
  writeURL();
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", boot);
}
