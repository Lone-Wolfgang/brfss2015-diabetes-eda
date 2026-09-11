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
    shape:     "#e5eaed",   // pale silhouette of the whole map, the same on every tab
    flag:      "#e2a52b",   // flagged: never screened, risk like the diagnosed
    ink:       "#22303c",
    rIn:       1.9,  aIn:   0.85,
    rDx:       2.0,  aDx:   0.95,
    rShape:    1.7,
    triSide:   8.5,         // flagged triangle, side length
    veil:      0.6,         // how far the map fades back while a respondent is selected
    fadeMs:    260,
    lineMs:    380,
    startTab:  "Everyone",
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
  const NB = new Uint16Array(bytes(D.nbrs).buffer);   // N x K
  const FLAGGED = [];
  for (let i = 0; i < N; i++) if (FLAG[i]) FLAGGED.push(i);

  // One rule on every tab: only the group is drawn. A pale silhouette of the whole map sits behind it
  // so the layout stays recognisable as you switch tabs.
  const layers = [{ group: "Overview", label: "Everyone", all: true, n: N, share: 100 }].concat(D.layers);
  const maskCache = {};
  const maskOf = (li) => (layers[li].all ? null : maskCache[li] || (maskCache[li] = unbits(layers[li].bits, N)));

  const flaggedIn = (li) => {             // how many flagged respondents a group contains
    const m = maskOf(li); if (!m) return FLAGGED.length;
    let c = 0; for (const i of FLAGGED) c += m[i]; return c;
  };

  const fmt = new Intl.NumberFormat("en-US");
  const pct = (v, d = 1) => (v == null ? "\u2013" : v.toFixed(d) + "%");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

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
    "<p>Pick a group and its members are drawn in colour: red if diagnosed with diabetes or prediabetes, blue if not. " +
    "Everyone else is removed, leaving a pale outline of the whole map, so you can see where the group sits and how " +
    "much red it carries.</p>" +
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
    " respondents (the held-out screened set plus everyone never screened)." +
    (D.repo_url ? ' <a href="' + esc(D.repo_url) + '">Notebook and code</a>.' : "");

  // ---------------------------------------------------------------- tabs
  const picker = $("picker"), tabs = [];
  let row = null, lastG = null;
  layers.forEach((L, i) => {
    if (L.group !== lastG) {
      row = document.createElement("div"); row.className = "grp";
      const gl = document.createElement("div"); gl.className = "grp-label"; gl.textContent = L.group;
      const bx = document.createElement("div"); bx.className = "grp-btns";
      row.append(gl, bx); picker.appendChild(row); lastG = L.group;
    }
    const b = document.createElement("button");
    b.className = "tab"; b.type = "button"; b.setAttribute("role", "tab");
    b.textContent = L.label; b.tabIndex = -1;
    b.addEventListener("click", () => setLayer(i));
    row.lastChild.appendChild(b); tabs.push(b);
  });
  picker.addEventListener("keydown", (e) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    setLayer((cur + d + layers.length) % layers.length);
    tabs[cur].focus();
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
  let cur = 0, showFlags = true, sel = -1;

  function drawBase() {
    const m = maskOf(cur), inG = (i) => !m || m[i];
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
  function caption() {
    const L = layers[cur], el = $("caption");
    let title = L.label, sub;
    if (L.all) sub = fmt.format(N) + " respondents, " + fmt.format(S.flagged) + " flagged";
    else {
      sub = fmt.format(L.n) + " people, " + pct(L.share, 0) + " of the map";
      if (L.dx_in != null) sub += ". Diagnosed among the screened: " + pct(L.dx_in) +
        (L.dx_out != null ? " in this group, " + pct(L.dx_out) + " outside" : "");
      else sub += ". None were screened, so a \u201cnot diagnosed\u201d label here is unverified";
    }
    el.innerHTML = '<div class="cap-title"></div><div class="cap-sub"></div>';
    el.firstChild.textContent = title;
    el.lastChild.textContent = sub;
  }

  function legend() {
    const L = layers[cur], who = L.all ? "" : L.label + ", ";
    const item = (sw, text) => '<span class="lg">' + sw + esc(text) + "</span>";
    const dotSw = (c) => '<i class="sw" style="background:' + c + '"></i>';
    let h = item(dotSw(LOOK.dx), L.all ? "Diagnosed" : who + "diagnosed") +
            item(dotSw(LOOK.nodx), L.all ? "Not diagnosed" : who + "not diagnosed");
    if (showFlags) {
      const nIn = flaggedIn(cur);
      if (L.all) h += item('<i class="sw tri"></i>', "Flagged: never screened, risk like the diagnosed");
      else if (nIn > 0) h += item('<i class="sw tri"></i>', "Flagged, in this group (" + fmt.format(nIn) + ")");
    }
    if (!L.all) h += item('<i class="sw shape"></i>', "Rest of the map (not shown)");
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
    const L = layers[cur], el = $("groupPanel");
    if (L.all) {
      el.innerHTML = "<h2>Everyone on the map</h2>" +
        stat("respondents", fmt.format(N)) +
        stat("screened (held-out test set)", fmt.format(S.n_test)) +
        stat("never screened", fmt.format(S.n_production + S.n_unscreened_pos)) +
        stat("flagged", fmt.format(S.flagged));
      return;
    }
    const nf = flaggedIn(cur);
    el.innerHTML = "<h2></h2>" +
      stat("in this group", fmt.format(L.n) + " (" + pct(L.share) + ")") +
      stat("never screened", pct(L.never_screened)) +
      stat("flagged", fmt.format(nf) + " of " + fmt.format(S.flagged)) +
      stat("diagnosed, screened in group", pct(L.dx_in)) +
      stat("diagnosed, screened outside", pct(L.dx_out)) +
      stat("neighbour \u03ba", L.kappa.toFixed(2)) +
      '<div class="kbar"><i style="width:' + Math.max(0, Math.min(1, L.kappa)) * 100 + '%"></i></div>' +
      '<p class="note">' + verdict(L.kappa) + "</p>";
    el.querySelector("h2").textContent = L.label;
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
      el.innerHTML = "<h3>Compare a respondent with their neighbours</h3>" +
        '<p class="note">Click anyone in ' + (layers[cur].all ? "the map" : "this group") + " to see the " + K +
        " people whose answers are closest to theirs. Their neighbours are shown wherever they are on the map.</p>" +
        '<div class="btn-row"><button type="button" class="btn primary" id="pickFlag">' +
        (pool().flagged ? "Show a flagged respondent" : "Show a respondent from this group") + "</button></div>";
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
  // The respondent view hangs off the group view: a selection always belongs to the current group,
  // and switching groups clears it.
  function setLayer(i, animateIt = true) {
    if (animateIt) { pctx.clearRect(0, 0, prev.width, prev.height); pctx.drawImage(base, 0, 0); }
    cur = i;
    sel = -1; personPanel();
    tabs.forEach((b, j) => { b.setAttribute("aria-selected", j === i ? "true" : "false"); b.tabIndex = j === i ? 0 : -1; });
    drawBase(); caption(); legend(); groupPanel();
    if (animateIt) play(true, false); else paint(1, 1);
  }
  function select(i) {
    sel = i; personPanel();
    if (i < 0) paint(1, 1); else play(false, true);
  }
  function pool() {                        // who the "show me" button draws from, within the group
    const m = maskOf(cur), fl = m ? FLAGGED.filter((i) => m[i]) : FLAGGED;
    if (fl.length) return { ids: fl, flagged: true };
    const ids = [];
    for (let i = 0; i < N; i++) if (!m || m[i]) ids.push(i);
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
    const m = maskOf(cur);
    for (let i = 0; i < N; i++) {
      if (m && !m[i]) continue;               // only members of the current group can be selected
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
  setLayer(Math.max(0, layers.findIndex((L) => L.label === LOOK.startTab)), false);
  personPanel();
  // for scripted screenshots and tests
  window.explorer = { setLayer, select, pickFlagged, layers, LOOK,
                      pos: (i) => [PX[i], PY[i]], selected: () => sel, inGroup: (i) => { const m = maskOf(cur); return !m || !!m[i]; } };
})();
