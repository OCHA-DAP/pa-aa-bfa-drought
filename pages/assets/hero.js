/* Animated particle-network hero.
 *
 * Same lineage as exploratory/paper/hdx-bg.html (the deck background) and the hero in
 * OCHA-DAP/ds-storm-impact-harmonisation, rescoped from a fixed full-viewport canvas to this
 * bounded header. Density and velocity are tuned down for a short panel.
 *
 * Decorative only: the canvas is aria-hidden and the page reads identically without it. */
(function () {
  var c = document.getElementById("bg");
  if (!c || !c.getContext) return;

  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var x = c.getContext("2d"), W, H, P;

  function reset() {
    var dpr = window.devicePixelRatio || 1, r = c.parentNode.getBoundingClientRect();
    W = r.width; H = r.height;
    if (!W || !H) return;
    c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    var n = Math.max(22, Math.min(80, Math.floor(W * H / 7000)));
    P = [];
    for (var i = 0; i < n; i++) P.push({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.7, vy: (Math.random() - 0.5) * 0.7
    });
  }

  function paint(step) {
    x.fillStyle = "#1e795f";                 /* --b6, matches the .hero background */
    x.fillRect(0, 0, W, H);
    if (step) {
      for (var i = 0; i < P.length; i++) {
        var p = P[i]; p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
      }
    }
    for (var i = 0; i < P.length; i++) {
      for (var j = i + 1; j < P.length; j++) {
        var a = P[i], b = P[j], dx = a.x - b.x, dy = a.y - b.y;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < 120) {
          x.strokeStyle = "rgba(255,255,255," + (1 - d / 120) * 0.45 + ")";
          x.lineWidth = 1; x.beginPath(); x.moveTo(a.x, a.y); x.lineTo(b.x, b.y); x.stroke();
        }
      }
    }
    x.fillStyle = "rgba(255,255,255,0.9)";
    for (var i = 0; i < P.length; i++) {
      x.beginPath(); x.arc(P[i].x, P[i].y, 1.9, 0, 6.283); x.fill();
    }
  }

  function frame() { paint(true); requestAnimationFrame(frame); }

  window.addEventListener("resize", function () { reset(); if (reduce) paint(false); });
  reset();
  if (reduce) { paint(false); } else { frame(); }
})();
