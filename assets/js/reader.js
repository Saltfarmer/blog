/* Reader features: theme toggle, progress bar, back to top, code labels, image zoom,
   copy section link, glossary tooltips, language filter. No dependencies. */
(function () {
  'use strict';

  var doc = document, root = doc.documentElement;
  var scriptSrc = (doc.currentScript && doc.currentScript.src) ||
    ((doc.querySelector('script[src*="reader.js"]') || {}).src || '');
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function ready(fn) {
    if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', fn); else fn();
  }

  function toast(msg) {
    var t = doc.querySelector('.bw-toast');
    if (!t) { t = doc.createElement('div'); t.className = 'bw-toast'; t.setAttribute('role', 'status'); doc.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(function () { t.classList.remove('show'); }, 1600);
  }

  /* ---------- Theme toggle ---------- */
  function applyTheme(dark) {
    root.setAttribute('data-theme', dark ? 'dark' : 'light');
    var link = doc.getElementById('dark-css');
    if (link) link.media = dark ? 'all' : 'not all';
    var m = doc.querySelector('meta[name="theme-color"]');
    if (m) m.setAttribute('content', dark ? '#071326' : '#0857C3');
  }
  function themeToggle() {
    var list = doc.querySelector('.greedy-nav .visible-links');
    if (!list) return;
    var li = doc.createElement('li');
    li.className = 'masthead__menu-item theme-toggle-item';
    var btn = doc.createElement('button');
    btn.type = 'button';
    btn.className = 'theme-toggle';
    li.appendChild(btn);
    list.appendChild(li);
    function paint() {
      var dark = root.getAttribute('data-theme') === 'dark';
      btn.innerHTML = '<i class="fas fa-' + (dark ? 'sun' : 'moon') + '" aria-hidden="true"></i>';
      btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
      btn.title = btn.getAttribute('aria-label');
    }
    btn.addEventListener('click', function () {
      var dark = root.getAttribute('data-theme') !== 'dark';
      applyTheme(dark);
      try { localStorage.setItem('bw-theme', dark ? 'dark' : 'light'); } catch (e) {}
      paint();
    });
    paint();
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var onChange = function (e) {
        var saved = null;
        try { saved = localStorage.getItem('bw-theme'); } catch (err) {}
        if (!saved) { applyTheme(e.matches); paint(); }
      };
      if (mq.addEventListener) mq.addEventListener('change', onChange);
    }
  }

  /* ---------- Reading progress + back to top ---------- */
  function progressAndTop() {
    var article = doc.querySelector('.layout--single .page__inner-wrap');
    var bar = null, fill = null;
    if (article) {
      bar = doc.createElement('div');
      bar.className = 'reading-progress';
      bar.setAttribute('aria-hidden', 'true');
      fill = doc.createElement('span');
      bar.appendChild(fill);
      doc.body.appendChild(bar);
    }
    var top = doc.createElement('button');
    top.type = 'button';
    top.className = 'to-top';
    top.setAttribute('aria-label', 'Back to top');
    top.innerHTML = '<i class="fas fa-arrow-up" aria-hidden="true"></i>';
    top.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
    });
    doc.body.appendChild(top);

    var ticking = false;
    function update() {
      ticking = false;
      var y = window.pageYOffset || root.scrollTop;
      top.classList.toggle('show', y > 600);
      if (fill) {
        var rect = article.getBoundingClientRect();
        var total = rect.height - window.innerHeight;
        var done = -rect.top;
        var p = total > 0 ? Math.min(1, Math.max(0, done / total)) : 0;
        fill.style.width = (p * 100).toFixed(1) + '%';
      }
    }
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  /* ---------- Code language labels ---------- */
  function codeLabels() {
    var skip = /^(plaintext|text|nohighlight|console)$/;
    Array.prototype.forEach.call(doc.querySelectorAll('div.highlighter-rouge'), function (el) {
      var m = el.className.match(/language-([\w+#-]+)/);
      if (m && !skip.test(m[1])) el.setAttribute('data-lang', m[1]);
    });
    Array.prototype.forEach.call(doc.querySelectorAll('figure.highlight'), function (el) {
      var c = el.querySelector('code[data-lang]');
      var l = c && c.getAttribute('data-lang');
      if (l && !skip.test(l)) el.setAttribute('data-lang', l);
    });
  }

  /* ---------- Image zoom ---------- */
  function lightbox() {
    var imgs = doc.querySelectorAll('.page__content img');
    Array.prototype.forEach.call(imgs, function (img) {
      if (img.closest('a') || img.closest('.toc') || img.classList.contains('emoji')) return;
      img.classList.add('zoomable');
    });
    function close() {
      var box = doc.querySelector('.lightbox');
      if (box) box.remove();
      doc.removeEventListener('keydown', onKey);
    }
    function onKey(e) { if (e.key === 'Escape') close(); }
    doc.addEventListener('click', function (e) {
      var img = e.target.closest && e.target.closest('img.zoomable');
      if (!img) return;
      if ((img.naturalWidth || 0) < 160) return;
      var box = doc.createElement('div');
      box.className = 'lightbox';
      box.setAttribute('role', 'dialog');
      box.setAttribute('aria-modal', 'true');
      box.setAttribute('aria-label', 'Image preview');
      var big = doc.createElement('img');
      big.src = img.currentSrc || img.src;
      big.alt = img.alt || '';
      var x = doc.createElement('button');
      x.type = 'button';
      x.setAttribute('aria-label', 'Close');
      x.innerHTML = '&times;';
      box.appendChild(big);
      box.appendChild(x);
      box.addEventListener('click', close);
      doc.body.appendChild(box);
      doc.addEventListener('keydown', onKey);
      x.focus();
    });
  }

  /* ---------- Copy link to section (uses the theme's # anchors) ---------- */
  function headerLinks() {
    doc.addEventListener('click', function (e) {
      var a = e.target.closest && e.target.closest('.header-link');
      if (!a) return;
      var href = a.getAttribute('href');
      if (!href || href.charAt(0) !== '#') return;
      var url = window.location.origin + window.location.pathname + href;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(url).then(function () { toast('Link to section copied'); });
      }
    });
  }

  /* ---------- Glossary tooltips (first mention per term) ---------- */
  function glossary() {
    var holder = doc.getElementById('glossary-data');
    var content = doc.querySelector('.layout--single .page__content');
    if (!holder || !content) return;
    var terms;
    try { terms = JSON.parse(holder.textContent); } catch (e) { return; }
    if (!terms || !terms.length) return;

    function esc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
    var entries = [];
    terms.forEach(function (t) {
      var names = [t.term].concat(t.aliases || []);
      names.forEach(function (n) { entries.push({ name: n, def: t.def, key: t.term.toLowerCase() }); });
    });
    entries.sort(function (a, b) { return b.name.length - a.name.length; });
    var re = new RegExp('(^|[^\\w])(' + entries.map(function (e) { return esc(e.name); }).join('|') + ')(?![\\w])', 'i');
    var lookup = {};
    entries.forEach(function (e) { lookup[e.name.toLowerCase()] = e; });

    var used = {};
    var blocked = 'a,code,pre,h1,h2,h3,h4,h5,h6,figcaption,.toc,aside,.highlight,script,style,textarea,button,.gloss,.series-nav,.takeaways,.runit,.subscribe,.page__related';
    var walker = doc.createTreeWalker(content, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        var p = node.parentElement;
        if (!p || p.closest(blocked)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [], n;
    while ((n = walker.nextNode())) nodes.push(n);

    nodes.forEach(function (node) {
      var text = node.nodeValue, guard = 0;
      var frag = null, last = 0, m;
      var rx = new RegExp(re.source, 'gi');
      while ((m = rx.exec(text)) && guard++ < 20) {
        var hit = lookup[m[2].toLowerCase()];
        if (!hit || used[hit.key]) continue;
        used[hit.key] = true;
        if (!frag) frag = doc.createDocumentFragment();
        var start = m.index + m[1].length;
        frag.appendChild(doc.createTextNode(text.slice(last, start)));
        var span = doc.createElement('span');
        span.className = 'gloss';
        span.tabIndex = 0;
        span.setAttribute('data-def', hit.def);
        span.textContent = m[2];
        frag.appendChild(span);
        last = start + m[2].length;
      }
      if (frag) {
        frag.appendChild(doc.createTextNode(text.slice(last)));
        node.parentNode.replaceChild(frag, node);
      }
    });
  }

  /* ---------- Language filter on post lists ---------- */
  function langFilter() {
    var items = Array.prototype.slice.call(doc.querySelectorAll('.archive .list__item[data-lang], .archive .grid__item[data-lang], .entries-list .list__item[data-lang]'));
    if (items.length < 2) return;
    var seen = {};
    items.forEach(function (it) { seen[it.getAttribute('data-lang')] = true; });
    if (Object.keys(seen).length < 2) return;

    var names = { en: 'English', id: 'Bahasa Indonesia' };
    var parents = [];
    items.forEach(function (it) { if (parents.indexOf(it.parentElement) < 0) parents.push(it.parentElement); });
    var originals = parents.map(function (p) { return Array.prototype.slice.call(p.children); });

    var bar = doc.createElement('div');
    bar.className = 'lang-filter';
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'Filter posts by language');
    bar.innerHTML = '<span class="lang-filter__label">Language:</span>';
    var buttons = [];
    ['all'].concat(Object.keys(seen).sort()).forEach(function (code) {
      var b = doc.createElement('button');
      b.type = 'button';
      b.textContent = code === 'all' ? 'All' : (names[code] || code);
      b.setAttribute('data-code', code);
      b.setAttribute('aria-pressed', code === 'all' ? 'true' : 'false');
      b.addEventListener('click', function () { apply(code); });
      bar.appendChild(b);
      buttons.push(b);
    });
    var first = parents[0];
    var anchor = first.previousElementSibling && first.previousElementSibling.classList.contains('archive__subtitle') ? first.previousElementSibling : first;
    anchor.parentNode.insertBefore(bar, anchor);

    function apply(code) {
      buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-code') === code)); });
      parents.forEach(function (p, i) {
        var keep = originals[i].filter(function (el) {
          var l = el.getAttribute && el.getAttribute('data-lang');
          return !l || code === 'all' || l === code;
        });
        while (p.firstChild) p.removeChild(p.firstChild);
        keep.forEach(function (el) { p.appendChild(el); });
        var prev = p.previousElementSibling;
        if (prev && prev.classList.contains('archive__subtitle')) {
          prev.hidden = !keep.some(function (el) { return el.getAttribute && el.getAttribute('data-lang'); });
        }
      });
    }
  }

  /* ---------- Sidebar toggle (wide screens) ---------- */
  function sidebarToggle() {
    var list = doc.querySelector('.greedy-nav .visible-links');
    if (!list || !doc.querySelector('.sidebar')) return;
    var li = doc.createElement('li');
    li.className = 'masthead__menu-item theme-toggle-item sidebar-toggle-item';
    var btn = doc.createElement('button');
    btn.type = 'button';
    btn.className = 'theme-toggle sidebar-toggle';
    btn.innerHTML = '<i class="fas fa-table-columns" aria-hidden="true"></i>';
    li.appendChild(btn);
    var theme = list.querySelector('.theme-toggle-item:not(.sidebar-toggle-item)');
    list.insertBefore(li, theme || null);
    function paint() {
      var hidden = root.classList.contains('sidebar-hidden');
      btn.setAttribute('aria-pressed', hidden ? 'true' : 'false');
      btn.setAttribute('aria-label', hidden ? 'Show sidebar' : 'Hide sidebar');
      btn.title = btn.getAttribute('aria-label');
    }
    btn.addEventListener('click', function () {
      var hidden = root.classList.toggle('sidebar-hidden');
      try { localStorage.setItem('bw-sidebar', hidden ? 'hidden' : 'shown'); } catch (e) {}
      paint();
    });
    paint();
  }

  /* ---------- Random post: dice in the top menu + any [data-random-post] button ---------- */
  var postUrls = null;
  function goRandom(btn) {
    if (btn) btn.classList.add('rolling');
    var go = function () {
      var here = location.pathname, read = readSet();
      var pool = postUrls.filter(function (u) { return u !== here && !read[u]; });
      if (!pool.length) pool = postUrls.filter(function (u) { return u !== here; }); // read everything? any post then
      if (pool.length) location.href = pool[Math.floor(Math.random() * pool.length)];
    };
    if (postUrls) return go();
    var base = scriptSrc.replace(/assets\/js\/reader\.js.*$/, '');
    fetch(base + 'posts.json').then(function (r) { return r.json(); })
      .then(function (d) { postUrls = d; go(); })
      .catch(function () { if (btn) btn.classList.remove('rolling'); toast('Could not load posts'); });
  }
  function randomPost() {
    var list = doc.querySelector('.greedy-nav .visible-links');
    if (list) {
      var li = doc.createElement('li');
      li.className = 'masthead__menu-item theme-toggle-item random-post-item';
      var btn = doc.createElement('button');
      btn.type = 'button';
      btn.className = 'theme-toggle random-post';
      btn.setAttribute('aria-label', 'Read a random post');
      btn.title = 'Random post';
      btn.innerHTML = '<i class="fas fa-dice" aria-hidden="true"></i>';
      li.appendChild(btn);
      list.insertBefore(li, list.querySelector('.theme-toggle-item') || null);
      btn.addEventListener('click', function () { goRandom(btn); });
    }
    doc.querySelectorAll('[data-random-post]').forEach(function (b) {
      b.addEventListener('click', function () { goRandom(b); });
    });
  }

  /* ---------- Reading memory (this browser only) ----------
     - remembers posts you finished  -> "Read" marks in lists, unread first in "Read next"
     - remembers a post you left halfway -> "Continue reading" on the home page */
  function store(key, val) {
    try { if (val === undefined) return JSON.parse(localStorage.getItem(key) || 'null'); localStorage.setItem(key, JSON.stringify(val)); } catch (e) { return null; }
  }
  function readSet() { return store('bw-read') || {}; }

  function readingMemory() {
    var read = readSet();
    var here = location.pathname;
    var content = doc.querySelector('.layout--single .page__content');
    var isPost = !!doc.querySelector('.read-next');

    // 1. Track progress on a post
    if (content && isPost) {
      var title = (doc.querySelector('.page__title') || {}).textContent || doc.title;
      title = title.trim();
      var saved = false, pending = false;
      var track = function () {
        pending = false;
        var r = content.getBoundingClientRect();
        var pct = Math.min(1, Math.max(0, (window.innerHeight - r.top) / r.height));
        if (pct >= 0.85 && !saved) {
          saved = true;
          read = readSet(); read[here] = Date.now(); store('bw-read', read);
          var last = store('bw-last'); if (last && last.u === here) { try { localStorage.removeItem('bw-last'); } catch (e) {} }
        } else if (!saved && pct > 0.12) {
          store('bw-last', { u: here, t: title, p: Math.round(pct * 100), y: Math.round(window.scrollY), ts: Date.now() });
        }
      };
      window.addEventListener('scroll', function () { if (!pending) { pending = true; requestAnimationFrame(track); } }, { passive: true });

      // Came back via "Continue reading"? jump to where they stopped
      if (location.hash === '#continue') {
        var last = store('bw-last');
        if (last && last.u === here) {
          history.replaceState(null, '', here);
          setTimeout(function () { window.scrollTo({ top: last.y, behavior: reduceMotion ? 'auto' : 'smooth' }); }, 150);
        }
      }

      // Read next: unread posts first
      var list = doc.querySelector('.read-next__list');
      if (list) {
        var items = Array.prototype.slice.call(list.children);
        items.sort(function (a, b) { return (read[a.dataset.url] ? 1 : 0) - (read[b.dataset.url] ? 1 : 0); });
        items.forEach(function (li) { if (read[li.dataset.url]) li.classList.add('is-read'); list.appendChild(li); });
      }
    }

    // 2. "Read" marks in post lists
    doc.querySelectorAll('.archive__item-title a[href]').forEach(function (a) {
      if (read[a.pathname]) a.closest('.archive__item').classList.add('is-read');
    });

    // 3. "Continue reading" on the home page (last 21 days)
    var last = store('bw-last');
    var home = doc.querySelector('.layout--home .archive, .layout--home #main .archive');
    if (home && last && last.u && Date.now() - last.ts < 21 * 864e5) {
      var box = doc.createElement('aside');
      box.className = 'continue-reading';
      box.innerHTML = '<span class="continue-reading__label"><i class="fas fa-book-open" aria-hidden="true"></i> Continue reading</span>' +
        '<a class="continue-reading__link"></a><span class="continue-reading__pct"></span>' +
        '<button type="button" class="continue-reading__x" aria-label="Dismiss">&times;</button>';
      var link = box.querySelector('a');
      link.href = last.u + '#continue';
      link.textContent = last.t;
      box.querySelector('.continue-reading__pct').textContent = last.p + '% read';
      box.querySelector('button').addEventListener('click', function () {
        try { localStorage.removeItem('bw-last'); } catch (e) {}
        box.remove();
      });
      home.insertBefore(box, home.firstChild);
    }
  }

  /* gc3k signature in the empty right column (wide screens).
     Only on list pages with a free right column (home, archives); posts keep the small one in the top bar. */
  function sideSig() {
    var main = doc.getElementById('main');
    if (!main || !window.matchMedia) return;
    var a = doc.createElement('a');
    a.className = 'side-sig';
    a.href = 'https://saltfarmer.github.io/';
    a.title = 'gc3k';
    a.setAttribute('aria-label', 'gc3k signature, go to the gate');
    // posts (anything with a table of contents or the wide post layout) never get the big signature
    if (main.querySelector('.sidebar__right, .page__content') || doc.body.classList.contains('wide')) return;
    var col = doc.createElement('div');
    col.className = 'side-sig-col';
    col.appendChild(a);
    main.appendChild(col);
    doc.documentElement.classList.add('has-side-sig');
  }

  ready(function () {
    sideSig();
    randomPost();
    readingMemory();
    sidebarToggle();
    themeToggle();
    progressAndTop();
    codeLabels();
    lightbox();
    headerLinks();
    glossary();
    langFilter();
  });
})();

/* ---------- Hide images that fail to load (404, blocked, expired links) ---------- */
(function () {
  'use strict';
  function lone(el, parent) {
    // true when `parent` contains nothing visible except `el` (ignoring whitespace and a caption)
    var kids = Array.prototype.filter.call(parent.childNodes, function (n) {
      if (n === el) return false;
      if (n.nodeType === 3) return n.nodeValue.trim() !== '';
      return n.nodeType === 1 && n.tagName !== 'BR' && n.tagName !== 'FIGCAPTION' && !n.classList.contains('img-missing');
    });
    return kids.length === 0;
  }
  function hide(img) {
    if (img.getAttribute('data-missing')) return;
    img.setAttribute('data-missing', '1');
    var target = img;
    var a = img.parentElement && img.parentElement.tagName === 'A' ? img.parentElement : null;
    if (a && lone(img, a)) target = a;
    var fig = target.closest ? target.closest('figure') : null;
    if (fig && lone(target, fig)) target = fig;
    var teaser = img.closest ? img.closest('.archive__item-teaser') : null;
    if (teaser) target = teaser;
    var p = target.parentElement;
    target.classList.add('img-missing');
    if (p && p.tagName === 'P' && lone(target, p)) p.classList.add('img-missing');
  }
  document.addEventListener('error', function (e) {
    var t = e.target;
    if (t && t.tagName === 'IMG' && !t.classList.contains('emoji')) hide(t);
  }, true);
  function scan() {
    Array.prototype.forEach.call(document.querySelectorAll('img'), function (img) {
      if (img.getAttribute('loading') === 'lazy') return; // lazy ones report errors when they load
      if (img.complete && img.naturalWidth === 0 && img.getAttribute('src')) hide(img);
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', scan); else scan();
  window.addEventListener('load', scan);
})();
