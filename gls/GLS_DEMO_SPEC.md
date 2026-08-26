# GLS Correlation Demo — Tech Spec

**This file is the full implementation contract.** It inlines every definition, formula, color, and edge case. An agent in `aabylay.github.io` must **not** need the paper repo, `gls_core`, or any relative path.

Same text lives in both places (keep them in sync if you edit):

- `/home/abylay/aabylay.github.io/gls/GLS_DEMO_SPEC.md` ← **open this repo and implement here**
- `/home/abylay/papers/260218_FANNS_VecDB_VLDB_workshop_CR/poster_upd/GLS_DEMO_SPEC.md`

## Links (embed these on the demo page)

Use real `<a href>` tags in the header and footer. Do not depend on local files.

| What | URL |
|---|---|
| **Paper (abstract)** | https://arxiv.org/abs/2602.11443 |
| **Paper (PDF)** | https://arxiv.org/pdf/2602.11443 |
| **Code / filtered ANN-Benchmarks** | https://github.com/aabylay/ANN-benchmark-HQ |
| **This demo** | https://aabylay.github.io/gls/ |
| **Author site** | https://aabylay.github.io/ |

Suggested footer copy:

> Amanbayev, Tsan, Dang, Rusu. *Filtered Approximate Nearest Neighbor Search in Vector Databases.* VLDB 2026 VecDB Workshop. [arXiv:2602.11443](https://arxiv.org/abs/2602.11443) · [PDF](https://arxiv.org/pdf/2602.11443) · [code](https://github.com/aabylay/ANN-benchmark-HQ)

Also add a Projects bullet on the homepage (`index.html`) pointing at `/gls/` and the same two links.

---

## 0. Paper excerpt (verbatim meaning)

GLS contrasts a filter’s prevalence in the **local neighborhood of a query** against its **global** prevalence.

For dataset \(\mathcal{D}\) of \(N\) vectors and filter \(\phi:\mathcal{D}\to\{0,1\}\) with **non-empty** support:

**Global selectivity** (corpus-level prior):

\[
\sigma_g = \frac{|\{v\in\mathcal{D}:\phi(v)=1\}|}{N} \in (0,1]
\]

Given query \(q\) with unfiltered \(k\)-nearest neighborhood \(\mathcal{N}_q\) (\(|\mathcal{N}_q|=k\)):

**Local selectivity:**

\[
\sigma_l = \frac{|\{v\in\mathcal{N}_q:\phi(v)=1\}|}{k} \in [0,1]
\]

\(\sigma_l > \sigma_g\): local **enrichment**. \(\sigma_l < \sigma_g\): local **depletion**.

**Selectivity ratio** \(r = \sigma_l / \sigma_g \in [0,\infty)\):

- \(r=1\) neutral
- \(r>1\) enrichment
- \(0 \le r < 1\) depletion

\(r\) is unbounded (bad for averaging / cost-model thresholds). Bilinear (Möbius) map:

\[
\rho_q = \frac{r-1}{r+1} \in [-1,1)
\]

- \(\rho_q=0\) independent / neutral
- \(\rho_q>0\) local enrichment (**easier** filtered queries)
- \(\rho_q<0\) local depletion (**harder** filtered queries)

Inverse: \(r = (1+\rho_q)/(1-\rho_q)\). Dataset mean \(\bar\rho = \frac{1}{|\mathcal{Q}|}\sum_q \rho_q\) exists in the paper; **v1 demo does not need \(\bar\rho\)**.

This demo uses **exact** \(k\)-NN, not the paper’s optional ANN / sample estimates.

**Finite-\(k\) artifact (paper):** at low \(\sigma_g\), a neighborhood of size \(k\) often contains fewer valid points than \(\sigma_g k\), so \(\rho_q\) drifts negative even when the filter is independent of geometry. Surface a one-line note when expected local pass count \(\sigma_g k \lesssim 3\).

**Worked numbers to match on first load (clustered, \(\sigma_g=0.2\)):** something in the ballpark of \(\sigma_l\approx 0.33\), \(r\approx 1.65\), \(\rho_q\approx 0.245\) if \(q\) sits in the valid blob. Exact values depend on the random draw; the **sign** must be clearly positive.

---

## 1. Goal

Mostly visual: \(N\) random points in a 2D unit box, a metadata filter, a query the user drops/drags. Live \(\sigma_g\), \(\sigma_l\), \(r\), \(\rho_q\).

Audience: poster-QR on a phone, plus paper readers. One screen, no login, no backend.

**Out of scope for v1:** ANN indexes, QPS/recall plots, MoReVec, FAISS/Milvus/pgvector, GLS-CorE estimator, ACORN distance-correlation panel, \(\bar\rho\) histogram.

---

## 2. Metric — implement exactly this

Dataset \(\mathcal{D}\): \(N\) points in \([0,1]^2\). Query \(q\in[0,1]^2\) is **not** inserted into \(\mathcal{D}\).

\(\mathcal{N}_q\): exact unfiltered \(k\)-NN of \(q\) in \(\mathcal{D}\), Euclidean \(L_2\). Distance = squared Euclidean is fine for ranking. Ties: smaller index wins.

```js
// stable rho; do not form r first when sigma_l === 0
function rhoFromSelectivities(sigmaL, sigmaG) {
  if (!(sigmaG > 0)) return null; // empty filter: r and rho are n/a
  return (sigmaL - sigmaG) / (sigmaL + sigmaG);
}
```

Equivalent to \((r-1)/(r+1)\) with \(r=\sigma_l/\sigma_g\). When \(\sigma_l=0\) and \(\sigma_g>0\), \(\rho_q=-1\).

| Case | Display |
|---|---|
| \(\sigma_g=0\) | counts \(0/N\), \(n_l/k\); \(r\) and \(\rho_q\) = `n/a` (never NaN/Inf) |
| \(\sigma_g=1\) | \(\sigma_l=1\), \(r=1\), \(\rho_q=0\) always |
| \(\sigma_l=0\), \(\sigma_g>0\) | \(r=0\), \(\rho_q=-1\) |

Always show **counts** \(n_g/N\) and \(n_l/k\) next to the ratios.

Pedagogical \(k\) (not the paper’s analysis \(k=2048\)):

\[
k = \max\bigl(1,\; \operatorname{round}(0.1\,N)\bigr)
\]

Default \(N=400\) \(\Rightarrow\) \(k=40\). Optional \(k\) slider \(1\ldots N\).

---

## 3. Filter–geometry coupling (required)

A selectivity slider **alone** is not enough. I.i.d. Bernoulli(\(\sigma_g\)) labels give \(\mathbb{E}[\sigma_l]=\sigma_g\), so \(\rho_q\approx 0\) everywhere and dragging \(q\) teaches nothing.

Each point has latent \(a_i\). Filter is a **rank threshold** (same spirit as paper scalar inequalities `attr ≥ t`):

```text
n_g = round(sigma_g * N)          // clamp to [0, N]
φ(i) = 1  iff  a_i is among the n_g largest a-values
```

Realized \(\sigma_g = n_g / N\) is exact. Slider is monotonic (raising \(\sigma_g\) only adds points).

**Geometry modes:**

| Mode | \(a_i\) | User should see |
|---|---|---|
| **Independent** | \(a_i \sim U[0,1]\), ignore \((x,y)\) | \(\rho_q\approx 0\) for any \(q\); noise grows as \(\sigma_g\) or \(k\) shrinks |
| **Clustered** (default) | \(a_i = \exp(-\|p_i-c\|^2 / (2s^2))\), then use raw score for ranking (no need to map to \([0,1]\)) | \(q\) **in** blob \(\to\) enrichment; \(q\) **outside** \(\to\) depletion |
| **Gradient** | \(a_i = x_i\) | \(\rho_q\) changes as \(q\) moves left/right |

Clustered defaults: blob center \(c=(0.35, 0.65)\), \(s=0.18\). First-load \(q\) near \(c\) so opening the page shows **positive** \(\rho_q\).

Default \(\sigma_g=0.20\).

---

## 4. Visuals

Draw a **dashed circle** through the \(k\)-th neighbor.

| Role | Fill | Stroke |
|---|---|---|
| Pass filter, not in \(k\)-NN | `#7dcc7d` | `#333` |
| Fail filter, not in \(k\)-NN | `#c8c8c8` | `#333` |
| In \(k\)-NN, fail filter | `#6a6ad4` | `#222` 1.5px |
| In \(k\)-NN **and** pass filter | split: left `#6a6ad4`, right `#7dcc7d` (or blue ring + green fill) | `#222` 1.5px |
| Query \(q\) | red star `#e05050` | `#222` |
| \(k\)-NN radius | none | dashed `#6a6ad4` |

Legend always visible. Page chrome (headers, sliders) may use UC Merced Bobcat Blue `#002856` and Gold `#DAA900`. Do **not** use those for data points.

**Live badge:**

- \(\rho_q > 0.05\): enrichment / easier
- \(\rho_q < -0.05\): depletion / harder
- else: neutral / independent
- \(\sigma_g=0\): skip badge

Show the four formulas with **current numbers** substituted, e.g. \(\rho_q=(1.65-1)/(1.65+1)=0.245\).

Short copy:

> GLS compares how often a filter fires **near the query** vs **in the whole set**. Counts, not distances — so local density does not fool it.

Header/footer must include the **Links** table URLs (paper abstract, PDF, GitHub).

---

## 5. Interaction

| Control | Spec |
|---|---|
| Plot | Unit square. Click/tap places \(q\). Drag \(q\) (or drag on empty space) moves \(q\). Live recompute. Clamp \(q\) to the box. |
| Filter selectivity | Slider \(\sigma_g\in[0,1]\), step \(0.01\). Default \(0.20\). |
| Geometry mode | Independent / Clustered / Gradient. |
| \(N\) | Slider \(100\)–\(1000\), step \(50\). Default \(400\). Changing \(N\) resamples points; keep \(q\) and \(\sigma_g\). |
| \(k\) | Show `k = round(0.1 N)`; optional slider \(1\ldots N\). |
| Resample points | New uniform \((x,y)\); rebuild \(a_i\) for current mode. |
| Reshuffle filter | New \(a_i\), same positions. |
| Reset | Defaults + new seed. |

Seeded RNG. Shareable query string:

```text
?n=400&sigma=0.2&mode=clustered&seed=7&qx=0.35&qy=0.65
```

Recompute on every pointer move (\(N\le 1000\): one pass of squared distances).

**Mobile:** single column, plot on top, controls below. Fat touch target for \(q\). No hover-only UI.

---

## 6. Stack

**Decision: GitHub Pages + vanilla HTML/CSS/JS + Canvas 2D. No bundler, no framework, no backend.**

This matches the existing site (`index.html` + `assets/css/styles.css`, served from `main`). The poster QR has to keep working for years with no npm/CI. \(N\le 1000\) exact \(k\)-NN is ~1 ms in JS, so there is nothing to offload.

| Layer | Choice | Why |
|---|---|---|
| Host | GitHub Pages, path `/gls/` | Poster already QRs https://aabylay.github.io/gls/ |
| Page | `gls/index.html` (+ optional `app.js` / `style.css`) | Three static files max |
| UI | Native `<input type="range">`, `<button>`, `<select>` | Sliders and modes; no component library |
| Layout | CSS Grid / Flex, mobile-first | Plot \| metrics on desktop; stacked on phone |
| Plot | **Canvas 2D** | Split-fill points, dashed \(k\)-NN circle, 60 fps drag of \(q\) |
| Input | Pointer Events on the canvas | Mouse + touch with one code path |
| Math | HTML + Unicode (`σ_g`, `ρ_q`) or a few `<sub>` tags | Only four formulas; skip KaTeX so hall Wi‑Fi is not required |
| RNG | Mulberry32 (or similar) seeded PRNG | Shareable `?seed=` |
| HiDPI | `canvas.width = cssSize * devicePixelRatio` | Sharp points on phones |

**Do not use:** React / Vite / Next, D3 / Plotly / Chart.js, Three.js, Pyodide, Tailwind/Bootstrap layout, a server, or Python in the browser. D3 would fight custom split fills and a draggable star. A bundler is a maintenance cost for a one-page explainer.

Homepage already loads Bootstrap Icons from a CDN for the profile page; the **demo must not require that CDN**. System font stack or the same local/CSS fonts as the homepage is fine. No required third-party JS.

Exact \(k\)-NN: one pass of squared Euclidean distances, `Float64Array`, take the \(k\) smallest (partial select is nice, full sort of 1000 is also fine).

```text
aabylay.github.io/
  index.html              # Projects bullet → /gls/
  gls/
    GLS_DEMO_SPEC.md
    index.html            # page + canvas + controls + paper/GitHub links
    app.js                # optional; all logic may live in index.html
    style.css             # optional
```

---

## 7. Acceptance checks

1. Independent + \(\sigma_g=0.5\) + \(q\) anywhere: \(\rho_q\) near 0 (noise \(\lesssim 0.15\) at \(N=400\), \(k=40\)).
2. Clustered: \(q\) in the blob \(\Rightarrow \rho_q>0\); opposite corner \(\Rightarrow \rho_q<0\).
3. \(\sigma_g=1\): \(\rho_q=0\) identically.
4. \(\sigma_g=0\): \(r\) and \(\rho_q\) are `n/a`; no NaN/Infinity.
5. \(\sigma_l=0\), \(\sigma_g>0\): \(\rho_q=-1\).
6. Dashed circle through the farthest of the \(k\) neighbors.
7. Split/blue-green count \(n_l\) matches the metric panel.
8. Phone-width layout; \(q\) placeable with a finger.
9. On-screen formulas match §0/§2, including \(\rho_q=(r-1)/(r+1)\).
10. After push to `main`, https://aabylay.github.io/gls/ serves the page.
11. Header or footer contains live links to https://arxiv.org/abs/2602.11443 (and/or the PDF) and https://github.com/aabylay/ANN-benchmark-HQ.

---

## 8. Where to run this spec

**Implement in `/home/abylay/aabylay.github.io`.** Create `gls/index.html`, push `main`.

Do not implement in the paper tree or in `/home/abylay/gls_core`.

You already have everything above. Do not open other repos for definitions.
