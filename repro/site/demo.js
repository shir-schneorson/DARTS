/* Scalar model of Section IV-E (eqs. scalar-model, rao-vs-lrt): H0 x|λ ~ N(0, λ), H1 x|λ ~ N(θ, λ).
   Every test decides H1 when x > aλ + b, with one global threshold set to P_FA = 1e-3 averaged over λ. */
(function () {
  "use strict";

  var N = 400;
  var PFA = 1e-3;
  var LAM_LO = 0.1;
  var LAM_HI = 10;
  /* SIGMA_HI tunes the test to every θ up to THETA_HI: 2κσσ² is 10.53 at σ = 5. */
  var THETA_HI = 10;
  var SIGMA_HI = 5;
  var X_HI = 20;
  var SVG_NS = "http://www.w3.org/2000/svg";

  /* λ is log-uniform on [LAM_LO, LAM_HI], represented by N equal-weight midpoints in log λ. */
  var lam = new Float64Array(N);
  var sd = new Float64Array(N);
  for (var i = 0; i < N; i++) {
    lam[i] = LAM_LO * Math.pow(LAM_HI / LAM_LO, (i + 0.5) / N);
    sd[i] = Math.sqrt(lam[i]);
  }

  /* Numerical Recipes erfcc: fractional error below 1.2e-7 for every real x. */
  function erfc(x) {
    var z = Math.abs(x);
    var t = 1 / (1 + 0.5 * z);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }

  function Q(z) {
    return 0.5 * erfc(z / Math.SQRT2);
  }

  /* P(x > aλ + b), averaged over λ, when x | λ ~ N(mu, λ). */
  function exceed(a, b, mu) {
    var s = 0;
    for (var i = 0; i < N; i++) s += Q((a * lam[i] + b - mu) / sd[i]);
    return s / N;
  }

  /* Bisection for f(k) = P_FA, f decreasing on [lo, hi]; stops early once the interval is exhausted. */
  function solve(f, lo, hi) {
    for (var it = 0; it < 100; it++) {
      var mid = 0.5 * (lo + hi);
      if (mid <= lo || mid >= hi) break;
      if (f(mid) > PFA) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  var cachedSigma = NaN;
  var cachedKappa = NaN;

  /* κσ of the smoothed Rao test x > κσ(λ + σ²); σ = 0 gives κ0 of the Rao test. */
  function kappaSmoothed(sigma) {
    if (sigma !== cachedSigma) {
      var s2 = sigma * sigma;
      cachedKappa = solve(function (k) { return exceed(k, k * s2, 0); }, 0, 1e3);
      cachedSigma = sigma;
    }
    return cachedKappa;
  }

  /* κ* of the most powerful test at θ, x > κ*λ + θ/2; κ* may be negative. */
  function kappaMP(theta) {
    return solve(function (k) { return exceed(k, theta / 2, 0); }, -1e3, 1e3);
  }

  function pdMP(theta) {
    return exceed(kappaMP(theta), theta / 2, theta);
  }

  /* The amplitude at which the smoothed test is most powerful: θσ = 2κσσ². */
  function thetaTuned(sigma) {
    return 2 * kappaSmoothed(sigma) * sigma * sigma;
  }

  /* The matched level: σ with 2κσσ² = θ, by bisection in log σ. */
  function matchedSigma(theta) {
    if (!(theta > 0)) return 0;
    var lo = Math.log(1e-4);
    var hi = Math.log(1e4);
    for (var it = 0; it < 60; it++) {
      var mid = 0.5 * (lo + hi);
      if (thetaTuned(Math.exp(mid)) < theta) lo = mid;
      else hi = mid;
    }
    return Math.exp(0.5 * (lo + hi));
  }

  function add(parent, tag, attrs, text) {
    var node = document.createElementNS(SVG_NS, tag);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    parent.appendChild(node);
    return node;
  }

  function f2(v) {
    return v.toFixed(2);
  }

  /* Sizes the viewBox to the rendered width, so 12px text stays 12px on screen. */
  function frame(svg) {
    var w = Math.round(svg.getBoundingClientRect().width) || 400;
    var h = Math.max(230, Math.min(320, Math.round(w * 0.72)));
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.textContent = "";
    return { w: w, h: h, l: 58, r: w - 14, t: 12, b: h - 48 };
  }

  /* Text built from [string, class] parts; class "sub" lowers the part as a subscript. */
  function label(parent, attrs, parts) {
    var node = add(parent, "text", attrs);
    var shift = 0;
    parts.forEach(function (p) {
      var dy = p[1] === "sub" ? 4 : 0;
      var span = add(node, "tspan", { dy: dy - shift }, p[0]);
      if (p[1]) span.setAttribute("class", p[1]);
      shift = dy;
    });
  }

  /* xt, yt: [pixel, label] ticks; an empty x label draws a minor gridline. */
  function axes(svg, f, xt, yt, xTitle, yTitle) {
    var g = add(svg, "g", { "aria-hidden": "true" });
    yt.forEach(function (t) {
      add(g, "line", { x1: f.l, x2: f.r, y1: t[0], y2: t[0], "class": "demo-grid" });
      add(g, "text", { x: f.l - 6, y: t[0] + 4, "text-anchor": "end" }, t[1]);
    });
    xt.forEach(function (t) {
      add(g, "line", { x1: t[0], x2: t[0], y1: f.t, y2: f.b, "class": t[1] ? "demo-grid" : "demo-grid demo-minor" });
      if (t[1]) add(g, "text", { x: t[0], y: f.b + 17, "text-anchor": "middle" }, t[1]);
    });
    add(g, "rect", { x: f.l, y: f.t, width: f.r - f.l, height: f.b - f.t, "class": "demo-frame" });
    label(g, { x: (f.l + f.r) / 2, y: f.h - 10, "text-anchor": "middle", "class": "demo-title" }, xTitle);
    label(g, {
      transform: "translate(18 " + (f.t + f.b) / 2 + ") rotate(-90)",
      "text-anchor": "middle",
      "class": "demo-title"
    }, yTitle);
  }

  function curve(g, cls, xs, ys) {
    var d = "";
    for (var j = 0; j < xs.length; j++) d += (j ? "L" : "M") + xs[j].toFixed(1) + " " + ys[j].toFixed(1);
    add(g, "path", { d: d, "class": cls });
  }

  /* Panel A: the three thresholds on x as functions of λ (log axis). */
  function drawBoundaries(svg, s) {
    var f = frame(svg);
    var X = function (l) { return f.l + ((Math.log10(l) + 1) / 2) * (f.r - f.l); };
    var Y = function (v) { return f.b - (v / X_HI) * (f.b - f.t); };
    var xt = [[X(0.1), "0.1"], [X(1), "1"], [X(10), "10"]];
    [0.1, 1].forEach(function (dec) {
      for (var m = 2; m < 10; m++) xt.push([X(m * dec), ""]);
    });
    var yt = [0, 5, 10, 15, 20].map(function (v) { return [Y(v), String(v)]; });
    axes(svg, f, xt, yt, [["λ", "it"], [" (local variance)"]], [["threshold on "], ["x", "it"]]);

    var clip = add(add(svg, "defs", {}), "clipPath", { id: "demo-clip-boundaries" });
    add(clip, "rect", { x: f.l, y: f.t, width: f.r - f.l, height: f.b - f.t });
    var g = add(svg, "g", { "clip-path": "url(#demo-clip-boundaries)", "aria-hidden": "true" });
    var ls = [];
    for (var j = 0; j <= 120; j++) ls.push(LAM_LO * Math.pow(LAM_HI / LAM_LO, j / 120));
    var px = ls.map(X);
    var line = function (cls, a, b) {
      curve(g, cls, px, ls.map(function (l) { return Y(a * l + b); }));
    };
    line("demo-rao", s.k0, 0);
    line("demo-smooth", s.ks, s.ks * s.sigma * s.sigma);
    line("demo-mp", s.km, s.theta / 2);
  }

  /* Panel B: P_D against θ for the Rao test, the smoothed test, and the most powerful envelope. */
  function drawPower(svg, s, base) {
    var f = frame(svg);
    var X = function (t) { return f.l + (t / THETA_HI) * (f.r - f.l); };
    var Y = function (p) { return f.b - p * (f.b - f.t); };
    var xt = [0, 2, 4, 6, 8, 10].map(function (v) { return [X(v), String(v)]; });
    var yt = [0, 0.2, 0.4, 0.6, 0.8, 1].map(function (v) { return [Y(v), v % 1 ? v.toFixed(1) : String(v)]; });
    axes(svg, f, xt, yt, [["target amplitude "], ["θ", "it"]],
      [["P", "it"], ["D", "sub"], [" at "], ["P", "it"], ["FA", "sub"], [" = 10⁻³"]]);

    var g = add(svg, "g", { "aria-hidden": "true" });
    var mx = X(s.theta);
    add(g, "line", { x1: mx, x2: mx, y1: f.t, y2: f.b, "class": "demo-marker" });
    var px = base.thetas.map(X);
    curve(g, "demo-rao", px, base.rao.map(Y));
    curve(g, "demo-smooth", px, s.smooth.map(Y));
    curve(g, "demo-mp", px, base.envelope.map(Y));
    if (s.sigma > 0 && s.tuned <= THETA_HI) {
      add(g, "circle", { cx: X(s.tuned), cy: Y(s.tunedPD), r: 5, "class": "demo-dot" });
    }
  }

  function init() {
    var root = document.getElementById("scalar-demo");
    if (!root) return;
    var ui = {
      sigma: document.getElementById("demo-sigma"),
      theta: document.getElementById("demo-theta"),
      sigmaValue: document.getElementById("demo-sigma-value"),
      thetaValue: document.getElementById("demo-theta-value"),
      match: document.getElementById("demo-match"),
      boundaries: document.getElementById("demo-boundaries"),
      power: document.getElementById("demo-power"),
      readout: document.getElementById("demo-readout")
    };

    var k0 = kappaSmoothed(0);
    var base = { thetas: [], rao: [], envelope: [] };
    for (var j = 0; j <= 160; j++) {
      var t = (THETA_HI * j) / 160;
      base.thetas.push(t);
      base.rao.push(exceed(k0, 0, t));
      base.envelope.push(pdMP(t));
    }

    var s = { sigma: parseFloat(ui.sigma.value), theta: parseFloat(ui.theta.value), k0: k0, smoothFor: NaN };
    var pending = false;

    function update() {
      pending = false;
      s.ks = kappaSmoothed(s.sigma);
      s.km = kappaMP(s.theta);
      var b = s.ks * s.sigma * s.sigma;
      if (s.sigma !== s.smoothFor) {
        s.smooth = base.thetas.map(function (t) { return exceed(s.ks, b, t); });
        s.smoothFor = s.sigma;
      }
      s.tuned = 2 * b;
      s.tunedPD = exceed(s.ks, b, s.tuned);
      drawBoundaries(ui.boundaries, s);
      drawPower(ui.power, s, base);

      ui.sigmaValue.textContent = f2(s.sigma);
      ui.thetaValue.textContent = f2(s.theta);
      ui.sigma.setAttribute("aria-valuetext", "σ = " + f2(s.sigma));
      ui.theta.setAttribute("aria-valuetext", "θ = " + f2(s.theta));

      var text = "<i>P</i><sub>D</sub> at <i>θ</i>&nbsp;=&nbsp;" + f2(s.theta) +
        ": Rao&nbsp;" + exceed(k0, 0, s.theta).toFixed(3) +
        " · smoothed&nbsp;" + exceed(s.ks, b, s.theta).toFixed(3) +
        " · most powerful&nbsp;" + exceed(s.km, s.theta / 2, s.theta).toFixed(3) + ". ";
      text += s.sigma > 0
        ? "The smoothed test is most powerful at <i>θ</i><sub><i>σ</i></sub>&nbsp;=&nbsp;" +
          "2<i>κ</i><sub><i>σ</i></sub><i>σ</i>²&nbsp;=&nbsp;" + f2(s.tuned) + "."
        : "At <i>σ</i>&nbsp;=&nbsp;0 the smoothed test is the Rao test, which is most powerful at no positive amplitude.";
      ui.readout.innerHTML = text;
    }

    function schedule() {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(update);
    }

    ui.sigma.addEventListener("input", function () {
      s.sigma = parseFloat(ui.sigma.value);
      schedule();
    });
    ui.theta.addEventListener("input", function () {
      s.theta = parseFloat(ui.theta.value);
      schedule();
    });
    ui.match.addEventListener("click", function () {
      var m = matchedSigma(s.theta);
      s.sigma = m;
      ui.sigma.value = String(m);
      schedule();
    });
    var panels = root.querySelector(".demo-panels") || root;
    var drawnWidth = panels.getBoundingClientRect().width;
    var onResize = function () {
      var w = panels.getBoundingClientRect().width;
      if (w !== drawnWidth) {
        drawnWidth = w;
        schedule();
      }
    };
    if (window.ResizeObserver) new ResizeObserver(onResize).observe(panels);
    else window.addEventListener("resize", onResize);
    update();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
