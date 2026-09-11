(function () {
  "use strict";
  const D = window.EXPLORER_DATA;
  const $ = (id) => document.getElementById(id);
  if (!D) {
    $("lede").textContent = "The map data did not load. Run scripts/export_explorer.py to create docs/data/explorer-data.js.";
    return;
  }

  // ---------------------------------------------------------------- look
  // One place to tune the appearance. Radii are in CSS pixels.
  const LOOK = {
    dx:        "#b5442e",   // in group, diagnosed
    nodx:      "#2c5d8f",   // in group, not diagnosed
    shape:     "#e5eaed",   // pale silhouette of the whole map, the same for every group
    flag:      "#e2a52b",   // flagged: never screened, risk like the diagnosed
    ink:       "#22303c",
    rIn:       1.9,  aIn:   0.85,
    rDx:       2.0,  aDx:   0.95,
    rShape:    1.7,
    triSide:   8.5,         // flagged triangle, side length
    veil:      0.6,         // how far the map fades back while a respondent is selected
    fadeMs:    260,
    lineMs:    380,
  };

  // ---------------------------------------------------------------- decode
  const bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const unbits = (s, n) => {
    const b = bytes(s), m = new Uint8Array(n);
    for (let j = 0; j < n; j++) m[j] = (b[j >> 3] >> (7 - (j & 7))) & 1;
    return m;
  };
  const N = D.n, K = D.k, NF = D.features.length, S = D.summary;
  const XY = new Uint16Array(bytes(D.xy).buffer);
  const DX = unbits(D.dx, N);
  const FLAG = unbits(D.flagged, N);
  const SPLIT = bytes(D.split);                       // 0 test, 1 production, 2 unscreened_pos
  const RISK = new Uint16Array(bytes(D.risk).buffer);  // x 10000
  const FV = bytes(D.fvals);                          // N x NF
  const NB = new Uint16Array(bytes(D.nbrs).buffer);   // N x K, nearest in feature space
  const FLAGGED = [];
  for (let i = 0; i < N; i++) if (FLAG[i]) FLAGGED.push(i);
  const COL = {};
  D.features.forEach((f, c) => { COL[f.key] = c; });
  const fv = (i, key) => FV[i * NF + COL[key]];

  const fmt = new Intl.NumberFormat("en-US");
  const pct = (v, d = 1) => (v == null ? "\u2013" : v.toFixed(d) + "%");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // ---------------------------------------------------------------- filters
  // Each question offers a few answers. Answers to the same question are combined with "or",
  // answers across questions with "and". `say` is how an answer reads in the group's title.
  const range = (key, lo, hi) => (i) => { const v = fv(i, key); return v >= lo && v <= hi; };
  const yesNo = (key, label, yes, no) => ({
    id: key, label, inline: true,
    opts: [{ label: "Yes", say: yes, test: (i) => fv(i, key) === 1 },
           { label: "No",  say: no,  test: (i) => fv(i, key) === 0 }],
  });
  const days = (key, label, noun) => ({
    id: key, label, suffix: " " + noun,
    opts: [{ label: "None", say: "0", test: range(key, 0, 0) },
           { label: "1\u201313", say: "1\u201313", test: range(key, 1, 13) },
           { label: "14 or more", say: "14+", test: range(key, 14, 30) }],
  });

  const SECTIONS = [
    { title: "Screening and diagnosis", open: true, qs: [
      { id: "screen", label: "Cholesterol screening", opts: [
        { label: "Screened", say: "screened", test: (i) => SPLIT[i] === 0 },
        { label: "Never screened", say: "never screened", test: (i) => SPLIT[i] !== 0 },
        { label: "Flagged", say: "flagged", test: (i) => FLAG[i] === 1,
          title: "Never screened, coded not diagnosed, with risk like the diagnosed" }] },
      { id: "dx", label: "Diabetes or prediabetes", opts: [
        { label: "Diagnosed", say: "diagnosed", test: (i) => DX[i] === 1 },
        { label: "Not diagnosed", say: "not diagnosed", test: (i) => DX[i] === 0 }] },
    ] },
    { title: "Demographics", open: true, qs: [
      { id: "Sex", label: "Sex", inline: true, opts: [
        { label: "Female", say: "female", test: (i) => fv(i, "Sex") === 0 },
        { label: "Male",   say: "male",   test: (i) => fv(i, "Sex") === 1 }] },
      { id: "Age", label: "Age", prefix: "aged ", opts: [
        { label: "18\u201339", say: "18\u201339", test: range("Age", 1, 4) },
        { label: "40\u201354", say: "40\u201354", test: range("Age", 5, 7) },
        { label: "55\u201364", say: "55\u201364", test: range("Age", 8, 9) },
        { label: "65\u201374", say: "65\u201374", test: range("Age", 10, 11) },
        { label: "75+",        say: "75+",        test: range("Age", 12, 13) }] },
      { id: "Education", label: "Education", opts: [
        { label: "Less than high school", say: "less than high school", test: range("Education", 1, 3) },
        { label: "High school",      say: "high school graduate", test: range("Education", 4, 4) },
        { label: "Some college",     say: "some college",         test: range("Education", 5, 5) },
        { label: "College graduate", say: "college graduate",     test: range("Education", 6, 6) }] },
      { id: "Income", label: "Household income", prefix: "income ", opts: [
        { label: "Under $15k",   say: "under $15k",   test: range("Income", 1, 2) },
        { label: "$15\u201325k", say: "$15\u201325k", test: range("Income", 3, 4) },
        { label: "$25\u201350k", say: "$25\u201350k", test: range("Income", 5, 6) },
        { label: "$50\u201375k", say: "$50\u201375k", test: range("Income", 7, 7) },
        { label: "$75k or more", say: "$75k or more", test: range("Income", 8, 8) }] },
    ] },
    { title: "Body and health", open: true, qs: [
      { id: "BMI", label: "BMI", note: "Under 18.5, 18.5\u201325, 25\u201330, 30 and over", opts: [
        { label: "Underweight", say: "underweight", test: range("BMI", 0, 18) },
        { label: "Normal",      say: "normal weight", test: range("BMI", 19, 24) },
        { label: "Overweight",  say: "overweight",  test: range("BMI", 25, 29) },
        { label: "Obese",       say: "obese",       test: range("BMI", 30, 255) }] },
      { id: "GenHlth", label: "General health", prefix: "in ", suffix: " health", opts:
        ["Excellent", "Very good", "Good", "Fair", "Poor"].map((l, j) =>
          ({ label: l, say: l.toLowerCase(), test: range("GenHlth", j + 1, j + 1) })) },
      days("PhysHlth", "Bad physical-health days, past 30", "bad physical days"),
      days("MentHlth", "Bad mental-health days, past 30", "bad mental days"),
    ] },
    { title: "Conditions", qs: [
      yesNo("HighBP", "High blood pressure", "high blood pressure", "no high blood pressure"),
      yesNo("HighChol", "High cholesterol", "high cholesterol", "no high cholesterol"),
      yesNo("HeartDiseaseorAttack", "Heart disease or attack", "heart disease", "no heart disease"),
      yesNo("Stroke", "Stroke", "had a stroke", "no stroke"),
      yesNo("DiffWalk", "Difficulty walking", "difficulty walking", "no difficulty walking"),
    ] },
    { title: "Behaviour", qs: [
      yesNo("Smoker", "Smoker", "smoker", "non-smoker"),
      yesNo("HvyAlcoholConsump", "Heavy drinker", "heavy drinker", "not a heavy drinker"),
      yesNo("PhysActivity", "Physically active", "physically active", "inactive"),
      yesNo("Fruits", "Fruit daily", "fruit daily", "no daily fruit"),
      yesNo("Veggies", "Vegetables daily", "vegetables daily", "no daily vegetables"),
    ] },
    { title: "Healthcare access", note: "Not used by the model or the map.", qs: [
      yesNo("AnyHealthcare", "Health coverage", "insured", "uninsured"),
      yesNo("NoDocbcCost", "Couldn\u2019t afford a doctor", "couldn\u2019t afford a doctor", "could afford a doctor"),
    ] },
  ];
  const QS = [];
  SECTIONS.forEach((sec) => sec.qs.forEach((q) => { q.sec = sec; QS.push(q); }));
  // one membership mask per answer, built once
  for (const q of QS) for (const o of q.opts) {
    o.mask = new Uint8Array(N);
    for (let i = 0; i < N; i++) o.mask[i] = o.test(i) ? 1 : 0;
  }

  const chosen = new Map();                 // question id -> Set of answer indices
  let MASK = null, COUNT = N;               // null means everyone

  function activeQs() {                     // questions that actually narrow the map
    return QS.filter((q) => chosen.has(q.id) && chosen.get(q.id).size < q.opts.length);
  }
  function computeMask() {
    const qs = activeQs();
    if (!qs.length) { MASK = null; COUNT = N; return; }
    const m = new Uint8Array(N).fill(1);
    for (const q of qs) {
      const os = [...chosen.get(q.id)].map((j) => q.opts[j].mask);
      for (let i = 0; i < N; i++) if (m[i]) {
        let hit = 0;
        for (const om of os) if (om[i]) { hit = 1; break; }
        m[i] = hit;
      }
    }
    let c = 0; for (let i = 0; i < N; i++) c += m[i];
    MASK = m; COUNT = c;
  }
  const inG = (i) => !MASK || MASK[i] === 1;

  function groupTitle() {
    const parts = activeQs().map((q) => {
      const says = [...chosen.get(q.id)].sort((a, b) => a - b).map((j) => q.opts[j].say);
      return (q.prefix || "") + says.join(" or ") + (q.suffix || "");
    });
    if (!parts.length) return "Everyone";
    const t = parts.join(", ");
    return t.charAt(0).toUpperCase() + t.slice(1);
  }

  // ---------------------------------------------------------------- map neighbours (for kappa)
  // 15 nearest points on the map, found with a uniform grid. x and y are put on the same scale first.
  const NBMAP = (function () {
    const xs = new Float64Array(N), ys = new Float64Array(N);
    for (let i = 0; i < N; i++) { xs[i] = XY[2 * i] / 65535; ys[i] = (XY[2 * i + 1] / 65535) * D.aspect; }
    const cell = Math.sqrt((4 * D.aspect) / N);
    const gx = Math.floor(1 / cell) + 1, gy = Math.floor(D.aspect / cell) + 1;
    const cx = new Int32Array(N), cy = new Int32Array(N), start = new Int32Array(gx * gy + 1), order = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      cx[i] = Math.min(gx - 1, Math.floor(xs[i] / cell)); cy[i] = Math.min(gy - 1, Math.floor(ys[i] / cell));
      start[cy[i] * gx + cx[i] + 1]++;
    }
    for (let c = 0; c < gx * gy; c++) start[c + 1] += start[c];
    const fill = start.slice(0, gx * gy);
    for (let i = 0; i < N; i++) order[fill[cy[i] * gx + cx[i]]++] = i;
    const out = new Uint16Array(N * K), bd = new Float64Array(K), bi = new Int32Array(K);
    for (let i = 0; i < N; i++) {
      let found = 0;
      const visit = (x, y) => {
        if (x < 0 || y < 0 || x >= gx || y >= gy) return;
        const c = y * gx + x;
        for (let p = start[c]; p < start[c + 1]; p++) {
          const j = order[p]; if (j === i) continue;
          const d = (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2;
          if (found === K && d >= bd[K - 1]) continue;
          let q = found < K ? found++ : K - 1;
          while (q > 0 && bd[q - 1] > d) { bd[q] = bd[q - 1]; bi[q] = bi[q - 1]; q--; }
          bd[q] = d; bi[q] = j;
        }
      };
      for (let r = 0; ; r++) {
        if (r === 0) visit(cx[i], cy[i]);
        else {
          for (let x = cx[i] - r; x <= cx[i] + r; x++) { visit(x, cy[i] - r); visit(x, cy[i] + r); }
          for (let y = cy[i] - r + 1; y <= cy[i] + r - 1; y++) { visit(cx[i] - r, y); visit(cx[i] + r, y); }
        }
        if (found === K && bd[K - 1] <= (r * cell) ** 2) break;
        if (r > gx + gy) break;
      }
      for (let k = 0; k < K; k++) out[i * K + k] = bi[k];
    }
    return out;
  })();

  function groupStats() {                   // everything the group panel and caption report
    const m = MASK;
    let nNever = 0, nFlag = 0, tin = 0, din = 0, tout = 0, dout = 0;
    for (let i = 0; i < N; i++) {
      const g = !m || m[i];
      if (g) { nNever += SPLIT[i] !== 0; nFlag += FLAG[i]; }
      if (SPLIT[i] === 0) { if (g) { tin++; din += DX[i]; } else { tout++; dout += DX[i]; } }
    }
    let kappa = null;
    if (m && COUNT > 0 && COUNT < N) {
      const p = COUNT / N, chance = p * p + (1 - p) * (1 - p);
      let agree = 0;
      for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) agree += m[i] === m[NBMAP[i * K + k]];
      kappa = (agree / (N * K) - chance) / (1 - chance);
    }
    return {
      n: COUNT, share: (100 * COUNT) / N, never: COUNT ? (100 * nNever) / COUNT : null, flagged: nFlag,
      dxIn: tin ? (100 * din) / tin : null, dxOut: tout ? (100 * dout) / tout : null, kappa,
    };
  }

  // ---------------------------------------------------------------- copy
  $("lede").innerHTML =
    "Each dot is one respondent to the 2015 BRFSS survey, placed near people who answered " +
    D.model_features.length + " health and demographic questions the same way. " +
    "Whether someone had been screened was left out of the map, so the " + fmt.format(S.n_production + S.n_unscreened_pos) +
    " people who had not had a cholesterol check in five years land wherever their answers put them. " +
    "<b>The " + fmt.format(S.flagged) + " gold triangles are never-screened respondents whose risk looks like the " +
    "diagnosed.</b> Most of them land among people who have a diagnosis. Click one to see who they resemble.";

  $("notes").innerHTML =
    "<div><h2>How to read the map</h2>" +
    "<p>Use the filters to pick a group. Its members are drawn in colour: red if diagnosed with diabetes or prediabetes, " +
    "blue if not. Everyone else is removed, leaving a pale outline of the whole map, so you can see where the group sits " +
    "and how much red it carries. Several answers to one question widen the group; answers to different questions " +
    "narrow it.</p>" +
    "<p>Clicking a point compares that person with the " + K + " respondents whose answers are closest to theirs on the " +
    D.model_features.length + " questions. Rows where most of those neighbours answered differently are marked in gold.</p></div>" +
    "<div><h2>What it shows</h2>" +
    "<p>Among the screened neighbours of flagged respondents, " + pct(S.nbr_dx_flagged) + " are diagnosed. Around people who are " +
    "diagnosed themselves the figure is " + pct(S.nbr_dx_diagnosed) + ", and around other never-screened respondents it is " +
    pct(S.nbr_dx_prod_other) + ". The flagged sit in the same neighbourhoods as the diagnosed.</p>" +
    "<p>Screening status barely shapes the layout: the never-screened group scores \u03ba = " + S.never_screened_kappa.toFixed(2) +
    " on the map, where 1 would mean they formed islands of their own.</p></div>" +
    "<div><h2>How it was built</h2>" +
    "<p>A gradient-boosted model was trained on " + fmt.format(S.n_train) + " screened respondents, without the three " +
    "healthcare-access questions, and tuned by cross-validation. On a held-out set of " + fmt.format(S.n_test) +
    " screened respondents it reaches AUC " + S.test_auc.toFixed(3) + ", with " + S.test_recorded_over_expected.toFixed(2) +
    " diagnoses recorded per diagnosis predicted.</p>" +
    "<p>Applied to the never-screened, it predicts " + fmt.format(Math.round(S.uns_expected)) + " diagnoses where " +
    fmt.format(S.uns_recorded) + " are recorded. One respondent is flagged per missing diagnosis, highest risk first " +
    "(risk \u2265 " + pct(100 * S.flag_threshold) + "). The survey has no blood test, so a flag marks an untrustworthy " +
    "negative label, not a confirmed case.</p></div>";

  $("footer").innerHTML = "BRFSS 2015 diabetes health indicators, CDC. Map: t-SNE on " + fmt.format(N) +
    " respondents (the held-out screened set plus everyone never screened).";

  // ---------------------------------------------------------------- sidebar
  const side = $("filters");
  side.innerHTML =
    '<div class="f-head">' +
      '<p class="f-count" id="fCount" aria-live="polite"></p>' +
      '<button type="button" class="btn small" id="fClear">Clear all</button>' +
    "</div>" +
    '<button type="button" class="btn f-toggle" id="fToggle" aria-expanded="false" aria-controls="fBody"></button>' +
    '<div class="f-body" id="fBody">' +
      '<p class="f-hint">Pick answers to narrow the map. Several answers to one question are combined with \u201cor\u201d.</p>' +
    "</div>";
  const body = $("fBody");
  const chipEls = new Map();                // question id -> [buttons]
  const badgeEls = new Map();               // section -> badge

  SECTIONS.forEach((sec) => {
    const det = document.createElement("details");
    det.className = "f-sec"; det.open = !!sec.open;
    const sum = document.createElement("summary");
    sum.innerHTML = '<span class="caret" aria-hidden="true"></span><span class="f-title"></span><span class="f-badge" hidden></span>';
    sum.querySelector(".f-title").textContent = sec.title;
    badgeEls.set(sec, sum.querySelector(".f-badge"));
    det.appendChild(sum);
    if (sec.note) {
      const p = document.createElement("p"); p.className = "f-note"; p.textContent = sec.note; det.appendChild(p);
    }
    sec.qs.forEach((q) => {
      const row = document.createElement("div");
      row.className = "f-q" + (q.inline ? " inline" : "");
      row.setAttribute("role", "group");
      const lab = document.createElement("div"); lab.className = "f-label"; lab.textContent = q.label;
      lab.id = "q-" + q.id; row.setAttribute("aria-labelledby", lab.id);
      const chips = document.createElement("div"); chips.className = "chips";
      const btns = q.opts.map((o, j) => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "chip"; b.textContent = o.label;
        b.setAttribute("aria-pressed", "false");
        if (o.title) b.title = o.title;
        b.addEventListener("click", () => toggle(q, j));
        chips.appendChild(b);
        return b;
      });
      chipEls.set(q.id, btns);
      row.append(lab, chips);
      if (q.note) { const n = document.createElement("div"); n.className = "f-note"; n.textContent = q.note; row.appendChild(n); }
      det.appendChild(row);
    });
    body.appendChild(det);
  });

  function toggle(q, j) {
    const s = chosen.get(q.id) || new Set();
    s.has(j) ? s.delete(j) : s.add(j);
    if (s.size) chosen.set(q.id, s); else chosen.delete(q.id);
    applyFilters();
  }
  function syncSidebar() {
    for (const q of QS) {
      const s = chosen.get(q.id);
      chipEls.get(q.id).forEach((b, j) => b.setAttribute("aria-pressed", s && s.has(j) ? "true" : "false"));
    }
    const act = activeQs();
    for (const [sec, el] of badgeEls) {
      const c = act.filter((q) => q.sec === sec).length;
      el.hidden = !c; el.textContent = c;
      el.setAttribute("aria-label", c + " active");
    }
    $("fCount").innerHTML = act.length
      ? "<b>" + fmt.format(COUNT) + "</b> of " + fmt.format(N) + " respondents"
      : "All <b>" + fmt.format(N) + "</b> respondents";
    $("fClear").disabled = !chosen.size;
    const t = $("fToggle");
    t.textContent = (side.classList.contains("open") ? "Hide filters" : "Show filters") + (act.length ? " (" + act.length + " active)" : "");
  }
  $("fClear").addEventListener("click", () => { chosen.clear(); applyFilters(); });
  $("fToggle").addEventListener("click", () => {
    const open = side.classList.toggle("open");
    $("fToggle").setAttribute("aria-expanded", open ? "true" : "false");
    syncSidebar();
  });

  // ---------------------------------------------------------------- sprites
  const cv = $("map"), ctx = cv.getContext("2d");
  const base = document.createElement("canvas"), bctx = base.getContext("2d");
  const prev = document.createElement("canvas"), pctx = prev.getContext("2d");
  const shape = document.createElement("canvas"), sctx = shape.getContext("2d");
  let dpr = 1, W = 0, H = 0;
  const PAD = 12;
  const PX = new Float32Array(N), PY = new Float32Array(N);

  const rgba = (hex, a) => {
    const v = parseInt(hex.slice(1), 16);
    return "rgba(" + (v >> 16) + "," + ((v >> 8) & 255) + "," + (v & 255) + "," + a + ")";
  };
  const sprites = {};
  function sprite(kind, size, fill, stroke, lw) {
    const key = [kind, size, fill, stroke, lw, dpr].join("|");
    if (sprites[key]) return sprites[key];
    const c = document.createElement("canvas");
    const s = Math.ceil((size * 2 + (lw || 0) * 2 + 2) * dpr);
    c.width = c.height = s;
    const g = c.getContext("2d"), m = s / 2;
    g.beginPath();
    if (kind === "dot") {
      g.arc(m, m, size * dpr, 0, 2 * Math.PI);
    } else {                                  // equilateral triangle, centred on its centroid
      const a = size * dpr, h = (a * Math.sqrt(3)) / 2;
      g.moveTo(m, m - (2 * h) / 3); g.lineTo(m + a / 2, m + h / 3); g.lineTo(m - a / 2, m + h / 3); g.closePath();
    }
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw * dpr; g.lineJoin = "round"; g.stroke(); }
    return (sprites[key] = c);
  }
  const stamp = (g, sp, i) => g.drawImage(sp, PX[i] * dpr - sp.width / 2, PY[i] * dpr - sp.height / 2);

  function layout() {
    dpr = window.devicePixelRatio || 1;
    W = cv.parentElement.clientWidth;
    H = Math.round(Math.min(W * D.aspect, Math.max(420, window.innerHeight * 0.86)));
    const sx = W - 2 * PAD, sy = H - 2 * PAD;
    const s = Math.min(sx, sy / D.aspect), ox = PAD + (sx - s) / 2, oy = PAD + (sy - s * D.aspect) / 2;
    for (let i = 0; i < N; i++) {
      PX[i] = ox + (XY[2 * i] / 65535) * s;
      PY[i] = oy + (1 - XY[2 * i + 1] / 65535) * s * D.aspect;
    }
    for (const c of [cv, base, prev, shape]) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    cv.style.height = H + "px";
    const sp = sprite("dot", LOOK.rShape, LOOK.shape);
    for (let i = 0; i < N; i++) stamp(sctx, sp, i);
  }

  // ---------------------------------------------------------------- drawing
  let showFlags = true, sel = -1;

  function drawBase() {
    bctx.clearRect(0, 0, base.width, base.height);
    bctx.drawImage(shape, 0, 0);
    const nodx = sprite("dot", LOOK.rIn, rgba(LOOK.nodx, LOOK.aIn));
    const dx   = sprite("dot", LOOK.rDx, rgba(LOOK.dx, LOOK.aDx));
    const skip = (i) => showFlags && FLAG[i];           // flagged are drawn as triangles instead
    for (let i = 0; i < N; i++) if (inG(i) && !DX[i] && !skip(i)) stamp(bctx, nodx, i);
    for (let i = 0; i < N; i++) if (inG(i) && DX[i]) stamp(bctx, dx, i);
    if (showFlags) {
      const full = sprite("tri", LOOK.triSide, LOOK.flag, LOOK.ink, 0.9);
      for (const i of FLAGGED) if (inG(i)) stamp(bctx, full, i);
    }
  }

  function paint(fade, line) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (fade < 1) { ctx.globalAlpha = 1 - fade; ctx.drawImage(prev, 0, 0); }
    ctx.globalAlpha = fade; ctx.drawImage(base, 0, 0); ctx.globalAlpha = 1;
    if (sel < 0) return;

    // a white veil pushes the whole map back so the selection and its neighbours stand out
    ctx.fillStyle = "rgba(255,255,255," + LOOK.veil * line + ")";
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.save(); ctx.scale(dpr, dpr);
    ctx.strokeStyle = rgba(LOOK.ink, 0.7); ctx.lineWidth = 1.1;
    ctx.beginPath();
    for (let k = 0; k < K; k++) {
      const j = NB[sel * K + k];
      ctx.moveTo(PX[sel], PY[sel]);
      ctx.lineTo(PX[sel] + (PX[j] - PX[sel]) * line, PY[sel] + (PY[j] - PY[sel]) * line);
    }
    ctx.stroke(); ctx.restore();
    if (line >= 1) for (let k = 0; k < K; k++) stamp(ctx, marker(NB[sel * K + k], 1), NB[sel * K + k]);
    stamp(ctx, sprite("dot", 12, "rgba(255,255,255,0.95)", LOOK.ink, 1.5), sel);
    stamp(ctx, marker(sel, 2.1), sel);
  }

  // a highlighted marker for the selection and its neighbours
  function marker(i, scale) {
    if (FLAG[i] && showFlags) return sprite("tri", LOOK.triSide * scale * 1.05, LOOK.flag, LOOK.ink, 1.3);
    return sprite("dot", 2.7 * scale, DX[i] ? LOOK.dx : LOOK.nodx, LOOK.ink, 1.2);
  }

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  const ease = (x) => 1 - Math.pow(1 - x, 3);
  let raf = 0, fadeStart = -1, lineStart = -1;
  function frame(now) {
    raf = 0;
    const f = fadeStart < 0 ? 1 : Math.min(1, (now - fadeStart) / LOOK.fadeMs);
    const l = lineStart < 0 ? 1 : Math.min(1, (now - lineStart) / LOOK.lineMs);
    paint(ease(f), ease(l));
    if (f < 1 || l < 1) raf = requestAnimationFrame(frame);
    else { fadeStart = -1; lineStart = -1; }
  }
  function play(fade, line) {
    if (reduced.matches) { fadeStart = lineStart = -1; return paint(1, 1); }
    const now = performance.now();
    if (fade) fadeStart = now;
    if (line) lineStart = now;
    if (!raf) raf = requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- caption and legend
  let ST = null;                             // stats for the current group

  function caption() {
    const el = $("caption");
    let sub;
    if (!MASK) sub = fmt.format(N) + " respondents, " + fmt.format(S.flagged) + " flagged";
    else if (!COUNT) sub = "No respondents match every filter. Remove one to widen the group.";
    else {
      sub = fmt.format(COUNT) + " people, " + pct(ST.share, ST.share < 1 ? 1 : 0) + " of the map";
      if (ST.dxIn != null) sub += ". Diagnosed among the screened: " + pct(ST.dxIn) +
        (ST.dxOut != null ? " in this group, " + pct(ST.dxOut) + " outside" : "");
      else sub += ". None were screened, so a \u201cnot diagnosed\u201d label here is unverified";
    }
    el.innerHTML = '<div class="cap-title"></div><div class="cap-sub"></div>';
    el.firstChild.textContent = groupTitle();
    el.lastChild.textContent = sub;
  }

  function legend() {
    const all = !MASK;
    const item = (sw, text) => '<span class="lg">' + sw + esc(text) + "</span>";
    const dotSw = (c) => '<i class="sw" style="background:' + c + '"></i>';
    let h = item(dotSw(LOOK.dx), all ? "Diagnosed" : "In group, diagnosed") +
            item(dotSw(LOOK.nodx), all ? "Not diagnosed" : "In group, not diagnosed");
    if (showFlags) {
      if (all) h += item('<i class="sw tri"></i>', "Flagged: never screened, risk like the diagnosed");
      else if (ST.flagged > 0) h += item('<i class="sw tri"></i>', "Flagged, in this group (" + fmt.format(ST.flagged) + ")");
    }
    if (!all) h += item('<i class="sw shape"></i>', "Rest of the map (not shown)");
    $("legend").innerHTML = h;
  }

  // ---------------------------------------------------------------- panels
  function verdict(k) {
    if (k >= 0.8) return "Defines islands: they are close to pure on this.";
    if (k >= 0.5) return "Shapes islands: most are dominated by one answer.";
    if (k >= 0.2) return "Some clumping, with many mixed islands.";
    return "Scattered: plays little part in the layout.";
  }
  const stat = (k, v) => '<div class="stat"><span>' + k + "</span><b>" + v + "</b></div>";

  function groupPanel() {
    const el = $("groupPanel");
    if (!MASK) {
      el.innerHTML = "<h2>Everyone on the map</h2>" +
        stat("respondents", fmt.format(N)) +
        stat("screened (held-out test set)", fmt.format(S.n_test)) +
        stat("never screened", fmt.format(S.n_production + S.n_unscreened_pos)) +
        stat("flagged", fmt.format(S.flagged));
      return;
    }
    if (!COUNT) {
      el.innerHTML = "<h2>No one matches</h2><p class=\"note\">These filters leave nobody on the map. " +
        "Remove one, or pick another answer to the same question.</p>";
      return;
    }
    el.innerHTML = "<h2>This group</h2>" +
      stat("in this group", fmt.format(ST.n) + " (" + pct(ST.share) + ")") +
      stat("never screened", pct(ST.never)) +
      stat("flagged", fmt.format(ST.flagged) + " of " + fmt.format(S.flagged)) +
      stat("diagnosed, screened in group", pct(ST.dxIn)) +
      stat("diagnosed, screened outside", pct(ST.dxOut)) +
      (ST.kappa == null ? "" :
        stat("neighbour \u03ba", ST.kappa.toFixed(2)) +
        '<div class="kbar"><i style="width:' + Math.max(0, Math.min(1, ST.kappa)) * 100 + '%"></i></div>' +
        '<p class="note">' + verdict(ST.kappa) + "</p>");
  }

  const AGE = ["18\u201324", "25\u201329", "30\u201334", "35\u201339", "40\u201344", "45\u201349", "50\u201354",
               "55\u201359", "60\u201364", "65\u201369", "70\u201374", "75\u201379", "80+"];
  const EDU = ["Never attended", "Elementary", "Some high school", "High school", "Some college", "College grad"];
  const INC = ["<$10k", "$10\u201315k", "$15\u201320k", "$20\u201325k", "$25\u201335k", "$35\u201350k", "$50\u201375k", "\u2265$75k"];
  const GEN = ["Excellent", "Very good", "Good", "Fair", "Poor"];
  const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

  function cell(kind, v, vals) {           // -> [this person, neighbours, differs from most neighbours]
    const share = mean(vals.map((x) => (x ? 1 : 0)));
    switch (kind) {
      case "bin": return [v ? "Yes" : "No", Math.round(100 * share) + "% yes", (v ? share : 1 - share) < 0.5];
      case "sex": return [v ? "Male" : "Female", Math.round(100 * share) + "% male", (v ? share : 1 - share) < 0.5];
      case "num": return [String(v), mean(vals).toFixed(1), false];
      case "genhlth": return [GEN[v - 1], GEN[Math.round(mean(vals)) - 1] + " (" + mean(vals).toFixed(1) + ")", false];
      case "age": return [AGE[v - 1], "median " + AGE[median(vals) - 1], false];
      case "edu": return [EDU[v - 1], "median " + EDU[median(vals) - 1], false];
      case "inc": return [INC[v - 1], "median " + INC[median(vals) - 1], false];
    }
    return [String(v), "", false];
  }

  function describe(i) {
    if (SPLIT[i] === 0) return DX[i] ? "Screened and diagnosed." : "Screened, not diagnosed.";
    if (SPLIT[i] === 2) return "Never screened, but diagnosed.";
    return FLAG[i] ? "Never screened, coded not diagnosed, and flagged." : "Never screened, coded not diagnosed.";
  }

  function personPanel() {
    const el = $("personPanel");
    if (sel < 0) {
      const p = pool();
      el.innerHTML = "<h3>Compare a respondent with their neighbours</h3>" +
        '<p class="note">Click anyone in ' + (MASK ? "this group" : "the map") + " to see the " + K +
        " people whose answers are closest to theirs. Their neighbours are shown wherever they are on the map.</p>" +
        '<div class="btn-row"><button type="button" class="btn primary" id="pickFlag"' + (p.ids.length ? "" : " disabled") + ">" +
        (p.flagged ? "Show a flagged respondent" : "Show a respondent from this group") + "</button></div>";
      $("pickFlag").onclick = pickFlagged;
      return;
    }
    const nb = Array.from(NB.subarray(sel * K, sel * K + K));
    const scr = nb.filter((j) => SPLIT[j] === 0), scrDx = scr.filter((j) => DX[j]);
    const allDx = nb.filter((j) => DX[j]).length;
    const dots = nb.map((j) => '<i class="' +
      (DX[j] ? "dx" : FLAG[j] ? "flag" : SPLIT[j] === 0 ? "nodx" : "uns") + '"></i>').join("");

    let rows = "", sepDone = false;
    D.features.forEach((f, c) => {
      if (!D.model_features.includes(f.key) && !sepDone) {
        rows += '<tr class="sep"><td colspan="3">Not used by the model or the map</td></tr>'; sepDone = true;
      }
      const [p, n, differs] = cell(f.kind, FV[sel * NF + c], nb.map((j) => FV[j * NF + c]));
      rows += "<tr" + (differs ? ' class="differs"' : "") + "><td>" + f.label + "</td><td>" + p + "</td><td>" + n + "</td></tr>";
    });

    el.innerHTML =
      "<h3>Selected respondent</h3>" +
      '<p class="who">' + describe(sel) +
      '<span class="risk">Modelled risk ' + pct(RISK[sel] / 100) + (SPLIT[sel] === 1
        ? " (flag threshold " + pct(100 * S.flag_threshold) + ")" : "") + "</span></p>" +
      '<div class="nbr-dots" aria-hidden="true">' + dots + "</div>" +
      '<p class="note">' + scr.length + " of the " + K + " nearest neighbours were screened, and " + scrDx.length +
      " of those are diagnosed" + (scr.length ? " (" + Math.round((100 * scrDx.length) / scr.length) + "%)" : "") +
      ". " + allDx + " of " + K + " are diagnosed in total. Hollow dots are never screened.</p>" +
      '<table class="cmp"><thead><tr><th>Question</th><th>This person</th><th>Neighbours</th></tr></thead><tbody>' +
      rows + "</tbody></table>" +
      '<div class="btn-row"><button type="button" class="btn primary" id="pickFlag">' +
      (pool().flagged ? "Show another flagged respondent" : "Show another respondent") + "</button>" +
      '<button type="button" class="btn" id="clearSel">Clear</button></div>';
    $("pickFlag").onclick = pickFlagged;
    $("clearSel").onclick = () => select(-1);
  }

  // ---------------------------------------------------------------- state
  // The respondent view hangs off the group: a selection always belongs to the current group,
  // and changing the filters clears it.
  function applyFilters(animateIt = true) {
    if (animateIt) { pctx.clearRect(0, 0, prev.width, prev.height); pctx.drawImage(base, 0, 0); }
    computeMask();
    ST = groupStats();
    sel = -1; personPanel();
    syncSidebar(); drawBase(); caption(); legend(); groupPanel();
    if (animateIt) play(true, false); else paint(1, 1);
  }
  function select(i) {
    sel = i; personPanel();
    if (i < 0) paint(1, 1); else play(false, true);
  }
  function pool() {                        // who the "show me" button draws from, within the group
    const fl = MASK ? FLAGGED.filter((i) => MASK[i]) : FLAGGED;
    if (fl.length) return { ids: fl, flagged: true };
    const ids = [];
    for (let i = 0; i < N; i++) if (inG(i)) ids.push(i);
    return { ids, flagged: false };
  }
  function pickFlagged() {
    const { ids } = pool();
    if (!ids.length) return;
    let i;
    do { i = ids[Math.floor(Math.random() * ids.length)]; } while (i === sel && ids.length > 1);
    select(i);
  }

  cv.addEventListener("click", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = -1, bd = 12 * 12;
    for (let i = 0; i < N; i++) {
      if (!inG(i)) continue;                  // only members of the current group can be selected
      const dx = PX[i] - x, dy = PY[i] - y, d = dx * dx + dy * dy - (FLAG[i] && showFlags ? 20 : 0);  // triangles are bigger targets
      if (d < bd) { bd = d; best = i; }
    }
    select(best);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && sel >= 0) select(-1); });
  $("showFlags").addEventListener("change", (e) => {
    showFlags = e.target.checked; drawBase(); legend(); paint(1, 1);
  });

  let rt = null;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { layout(); drawBase(); paint(1, 1); }, 150);
  });

  layout();
  applyFilters(false);
  // for scripted screenshots and tests: setFilters({ Sex: ["Male"], BMI: ["Obese"] })
  function setFilters(spec) {
    chosen.clear();
    for (const [id, labels] of Object.entries(spec || {})) {
      const q = QS.find((x) => x.id === id); if (!q) continue;
      const s = new Set(labels.map((l) => q.opts.findIndex((o) => o.label === l)).filter((j) => j >= 0));
      if (s.size) chosen.set(id, s);
    }
    applyFilters();
  }
  window.explorer = { setFilters, select, pickFlagged, questions: QS, LOOK, stats: () => ST,
                      pos: (i) => [PX[i], PY[i]], selected: () => sel, inGroup: inG };
})();
