// Light and dark theme, switched by a camera aperture: stopped down for dark,
// wide open for light. The choice is remembered on this browser; without one
// the page follows the system setting. The new theme is revealed from the
// button outwards, like an iris opening.
(function () {
  var KEY = 'fiducia-site-theme';
  var root = document.documentElement;
  var button = document.getElementById('theme-toggle');
  if (!button) return;
  var svg = button.querySelector('svg');
  var system = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var NS = 'http://www.w3.org/2000/svg';
  var RING = 10, OPEN = 6.6, SHUT = 2.4;

  function make(tag, attrs) {
    var node = document.createElementNS(NS, tag);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    svg.appendChild(node);
    return node;
  }
  make('circle', { cx: 0, cy: 0, r: RING, 'class': 'aperture__ring' });
  var blades = [];
  for (var i = 0; i < 6; i++) blades.push(make('line', { 'class': 'aperture__blade' }));
  var dot = make('circle', { cx: 0, cy: 0, r: 1.5, 'class': 'aperture__dot' });

  // Six blade edges around a hexagonal opening of radius a, each carried on
  // to the lens barrel, as an iris is usually drawn. Closing turns it a little.
  function draw(a) {
    var twist = (OPEN - a) * 9 * Math.PI / 180, pts = [];
    for (var k = 0; k < 6; k++) {
      var t = k * Math.PI / 3 + twist;
      pts.push([a * Math.cos(t), a * Math.sin(t)]);
    }
    for (k = 0; k < 6; k++) {
      var p = pts[k], q = pts[(k + 1) % 6];
      var dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy);
      dx /= len; dy /= len;
      var b = p[0] * dx + p[1] * dy, c = p[0] * p[0] + p[1] * p[1] - RING * RING;
      var s = -b + Math.sqrt(b * b - c);
      blades[k].setAttribute('x1', p[0].toFixed(2));
      blades[k].setAttribute('y1', p[1].toFixed(2));
      blades[k].setAttribute('x2', (p[0] + s * dx).toFixed(2));
      blades[k].setAttribute('y2', (p[1] + s * dy).toFixed(2));
    }
    dot.setAttribute('r', (0.6 + 0.32 * a).toFixed(2));
  }

  var opening = null, frame = 0;
  function tween(target) {
    cancelAnimationFrame(frame);
    if (opening === null || reduce) { opening = target; draw(target); return; }
    var from = opening, start = null;
    (function step(now) {
      if (start === null) start = now;
      var k = Math.min(1, (now - start) / 520);
      k = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      opening = from + (target - from) * k;
      draw(opening);
      if (k < 1) frame = requestAnimationFrame(step);
    })(performance.now());
  }

  function current() {
    return root.getAttribute('data-theme') || (system && system.matches ? 'dark' : 'light');
  }
  function sync() {
    var dark = current() === 'dark';
    button.setAttribute('aria-label', dark ? 'Switch to the light theme' : 'Switch to the dark theme');
    button.title = dark ? 'Open the aperture: light theme' : 'Stop down: dark theme';
    tween(dark ? SHUT : OPEN);
  }

  button.addEventListener('click', function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    function apply() {
      root.setAttribute('data-theme', next);
      try { localStorage.setItem(KEY, next); } catch (e) { /* storage off: this visit only */ }
      sync();
      window.dispatchEvent(new Event('fiducia:theme'));
    }
    if (!document.startViewTransition || reduce) { apply(); return; }
    var r = button.getBoundingClientRect();
    var x = r.left + r.width / 2, y = r.top + r.height / 2;
    var radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    var transition = document.startViewTransition(apply);
    transition.ready.then(function () {
      root.animate(
        { clipPath: ['circle(0px at ' + x + 'px ' + y + 'px)', 'circle(' + radius + 'px at ' + x + 'px ' + y + 'px)'] },
        { duration: 700, easing: 'cubic-bezier(0.65, 0, 0.35, 1)', pseudoElement: '::view-transition-new(root)' }
      );
    }).catch(function () { /* the browser skipped the animation; the theme has still changed */ });
    transition.finished.catch(function () {});
  });

  if (system && system.addEventListener) {
    system.addEventListener('change', function () { if (!root.getAttribute('data-theme')) sync(); });
  }
  sync();
})();
