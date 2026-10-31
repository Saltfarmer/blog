/* Motion and small delights. Skipped entirely when the visitor prefers reduced motion. */
(function () {
  'use strict';
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  var doc = document, root = doc.documentElement;
  var finePointer = window.matchMedia && window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  var COLORS = ['#0857C3', '#307FE2', '#71C5E8', '#D6EFFB', '#C37408'];

  function ready(fn) { if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', fn); else fn(); }

  /* ---------- Confetti / sparkles ---------- */
  var canvas = null, ctx = null, parts = [], raf = 0;
  function ensureCanvas() {
    if (canvas) return;
    canvas = doc.createElement('canvas');
    canvas.className = 'bw-confetti';
    canvas.setAttribute('aria-hidden', 'true');
    doc.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
    size();
    window.addEventListener('resize', size);
  }
  function size() {
    if (!canvas) return;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function burst(x, y, count, spread) {
    ensureCanvas();
    for (var i = 0; i < count; i++) {
      var a = -Math.PI / 2 + (Math.random() - 0.5) * (spread || 2.4);
      var s = 4 + Math.random() * 7;
      parts.push({
        x: x, y: y, vx: Math.cos(a) * s, vy: Math.sin(a) * s,
        r: 2 + Math.random() * 3, rot: Math.random() * 6, vr: (Math.random() - 0.5) * 0.4,
        c: COLORS[(Math.random() * COLORS.length) | 0], life: 0, max: 70 + Math.random() * 40,
        round: Math.random() < 0.5
      });
    }
    if (!raf) raf = requestAnimationFrame(tick);
  }
  function tick() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    parts = parts.filter(function (p) { return p.life < p.max && p.y < window.innerHeight + 20; });
    parts.forEach(function (p) {
      p.life++; p.vy += 0.28; p.vx *= 0.985; p.x += p.vx; p.y += p.vy; p.rot += p.vr;
      ctx.save();
      ctx.globalAlpha = Math.max(0, 1 - p.life / p.max);
      ctx.translate(p.x, p.y); ctx.rotate(p.rot); ctx.fillStyle = p.c;
      if (p.round) { ctx.beginPath(); ctx.arc(0, 0, p.r, 0, 6.2832); ctx.fill(); }
      else ctx.fillRect(-p.r, -p.r / 2, p.r * 2, p.r);
      ctx.restore();
    });
    raf = parts.length ? requestAnimationFrame(tick) : 0;
    if (!parts.length) ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  }

  /* ---------- Scroll reveal ---------- */
  function reveal() {
    var sel = [
      '.archive__subtitle', '.grid__item', '.list__item', '.lang-filter',
      '.page__content > h2', '.page__content > h3', '.page__content > figure', '.page__content > .highlighter-rouge',
      '.page__content > blockquote', '.page__content > table', '.page__content > .notice',
      '.takeaways', '.runit', '.subscribe', '.page__related-title', '.series-nav--compact'
    ].join(',');
    var els = Array.prototype.slice.call(doc.querySelectorAll(sel));
    if (!els.length) return;
    var vh = window.innerHeight, order = 0, below = [];
    els.forEach(function (el) {
      var top = el.getBoundingClientRect().top;
      if (top < vh) {
        // already on screen at load: a quick staggered entrance
        if (el.animate) el.animate(
          [{ opacity: 0, transform: 'translateY(14px)' }, { opacity: 1, transform: 'none' }],
          { duration: 520, delay: Math.min(order++, 8) * 70, easing: 'cubic-bezier(0.2,0.7,0.2,1)', fill: 'backwards' });
      } else {
        el.classList.add('reveal');
        if (el.classList.contains('grid__item')) el.style.setProperty('--d', ((below.length % 4) * 70) + 'ms');
        below.push(el);
      }
    });
    if (!below.length) return;
    if (!('IntersectionObserver' in window)) { below.forEach(function (el) { el.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -4% 0px' });
    below.forEach(function (el) { io.observe(el); });
    setTimeout(function () { below.forEach(function (el) { el.classList.add('in'); }); }, 6000); // safety net
  }

  /* ---------- Cards follow the pointer ---------- */
  function cards() {
    if (!finePointer) return;
    var active = null, pending = false, last = null;
    function apply() {
      pending = false;
      if (!active || !last) return;
      var r = active.getBoundingClientRect();
      var px = (last.clientX - r.left) / r.width, py = (last.clientY - r.top) / r.height;
      active.style.setProperty('--mx', (px * 100).toFixed(1) + '%');
      active.style.setProperty('--my', (py * 100).toFixed(1) + '%');
      active.style.setProperty('--ry', ((px - 0.5) * 7).toFixed(2) + 'deg');
      active.style.setProperty('--rx', ((0.5 - py) * 5).toFixed(2) + 'deg');
    }
    function reset(el) {
      el.style.removeProperty('--rx'); el.style.removeProperty('--ry');
    }
    doc.addEventListener('pointermove', function (e) {
      var c = e.target.closest ? e.target.closest('.grid__item .archive__item') : null;
      if (active && active !== c) { reset(active); }
      active = c; last = e;
      if (c && !pending) { pending = true; requestAnimationFrame(apply); }
    }, { passive: true });
    doc.addEventListener('pointerleave', function () { if (active) { reset(active); active = null; } }, true);
  }

  /* ---------- Logo: spin and sparkle ---------- */
  function logo() {
    var a = doc.querySelector('.site-logo');
    if (!a) return;
    var img = a.querySelector('img');
    a.addEventListener('click', function (e) {
      var r = (img || a).getBoundingClientRect();
      var here = a.pathname.replace(/\/$/, '') === window.location.pathname.replace(/\/$/, '');
      if (img) { img.classList.remove('spin'); void img.offsetWidth; img.classList.add('spin'); }
      burst(r.left + r.width / 2, r.top + r.height / 2, 22, 5);
      if (here) e.preventDefault();
    });
  }

  /* ---------- Theme switch: let colours glide ---------- */
  function themeGlide() {
    doc.addEventListener('click', function (e) {
      if (!e.target.closest || !e.target.closest('.theme-toggle')) return;
      root.classList.add('theme-anim');
      clearTimeout(themeGlide.t);
      themeGlide.t = setTimeout(function () { root.classList.remove('theme-anim'); }, 600);
    }, true);
  }

  /* ---------- Reached the end of a long post ---------- */
  function finish() {
    var content = doc.querySelector('.layout--single .page__content');
    var foot = doc.querySelector('.layout--single footer.page__meta');
    if (!content || !foot || content.offsetHeight < 1600 || !('IntersectionObserver' in window)) return;
    var done = false;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && !done && window.scrollY > 600) {
          done = true; io.disconnect();
          burst(window.innerWidth / 2, window.innerHeight * 0.78, 42, 2.6);
        }
      });
    }, { threshold: 0.9 });
    io.observe(foot);
  }

  ready(function () {
    reveal();
    cards();
    logo();
    themeGlide();
    finish();
  });
})();
