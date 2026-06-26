/* ============================================================
   deck-hud.js — orientation + intuitive navigation layer.
   Sits on top of <deck-stage> (the engine is untouched).

   Adds:
     • a fixed top progress bar whose 14 segments are coloured by
       each slide's source track — so the whole talk is a map and
       the PAPER (teal) slides are visible at a glance;
     • a chapter pill (name · Ch N · position) that fades when idle;
     • a first-slide "← → · click · scroll to move" hint;
     • wheel + click-to-advance navigation (drives the engine's own
       keyboard handler, so skip/clamp logic is reused);
     • count-up for [data-count] stats on slide enter.
   Honors prefers-reduced-motion and presentation mode.
   ============================================================ */
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;
  let slides = [], hud, metaEl, chDot, chName, chNum, chPos, hint;
  let idleTimer = null, presenting = false, lastIndex = -1;
  let wheelLock = false, wheelAccum = 0;

  const deck = () => document.querySelector('deck-stage');

  function collect() {
    slides = Array.from(document.querySelectorAll('deck-stage > section'));
  }

  function build() {
    collect();
    if (!slides.length) return false;

    hud = document.createElement('div');
    hud.className = 'deck-hud';

    const bar = document.createElement('div');
    bar.className = 'bar';
    slides.forEach((s) => {
      const seg = document.createElement('span');
      seg.className = 'seg';
      seg.setAttribute('data-track', s.dataset.track || 'context');
      const fill = document.createElement('span');
      fill.className = 'fill';
      seg.appendChild(fill);
      bar.appendChild(seg);
    });

    metaEl = document.createElement('div');
    metaEl.className = 'meta';
    metaEl.innerHTML =
      '<span class="ch-dot"></span>' +
      '<span class="ch-name"></span>' +
      '<span class="ch-num"></span>' +
      '<span class="ch-pos"></span>';

    hud.appendChild(bar);
    hud.appendChild(metaEl);
    document.body.appendChild(hud);

    chDot = metaEl.querySelector('.ch-dot');
    chName = metaEl.querySelector('.ch-name');
    chNum = metaEl.querySelector('.ch-num');
    chPos = metaEl.querySelector('.ch-pos');
    requestAnimationFrame(() => hud.setAttribute('data-ready', ''));

    hint = document.createElement('div');
    hint.className = 'advance-hint';
    hint.innerHTML =
      '<span class="keys"><span class="k">&larr;</span><span class="k">&rarr;</span></span>' +
      '<span>or click &middot; scroll to move <span class="arrow-anim">&rarr;</span></span>';
    document.body.appendChild(hint);
    return true;
  }

  function update(index) {
    if (!slides.length || index < 0) return;
    const fills = hud.querySelectorAll('.seg .fill');
    fills.forEach((f, i) => { f.style.width = i <= index ? '100%' : '0'; });

    const s = slides[index];
    if (!s) return;
    metaEl.setAttribute('data-track', s.dataset.track || 'context');
    const ch = s.dataset.chapter || '0';
    chName.textContent = s.dataset.chapterName || '';
    chNum.textContent = (ch && ch !== '0') ? ('Ch ' + ch) : '';
    chPos.textContent = (index + 1) + ' / ' + slides.length;

    if (lastIndex !== index) {
      s.querySelectorAll('[data-count]').forEach(runCount);
    }
    lastIndex = index;
    showMeta();

    if (index === 0 && !hint.dataset.dismissed) {
      setTimeout(() => { if (lastIndex === 0) hint.setAttribute('data-show', ''); }, 900);
    } else if (hint) {
      hint.removeAttribute('data-show');
      hint.dataset.dismissed = '1';
    }
  }

  function showMeta() {
    if (presenting || !metaEl) return;
    metaEl.setAttribute('data-show', '');
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => metaEl && metaEl.removeAttribute('data-show'), 2400);
  }

  function runCount(el) {
    const target = parseFloat(el.dataset.count);
    if (isNaN(target)) return;
    const suffix = el.dataset.suffix || '';
    if (reduce) { el.textContent = target + suffix; return; }
    const dur = 1000, t0 = performance.now();
    function tick(t) {
      const p = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * e) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function nav(dir) {
    window.dispatchEvent(new KeyboardEvent('keydown', {
      key: dir > 0 ? 'ArrowRight' : 'ArrowLeft', bubbles: true,
    }));
  }

  function onWheel(e) {
    if (presenting) return;
    if (Math.abs(e.deltaY) < 8) return;
    wheelAccum += e.deltaY;
    if (wheelLock) return;
    if (Math.abs(wheelAccum) > 60) {
      nav(wheelAccum > 0 ? 1 : -1);
      wheelAccum = 0;
      wheelLock = true;
      setTimeout(() => { wheelLock = false; }, 680);
    }
  }

  function onClick(e) {
    if (!finePointer) return;            // touch uses the engine's tap zones
    if (e.button !== 0) return;
    if (window.getSelection && String(window.getSelection())) return; // mid text-selection
    const path = e.composedPath ? e.composedPath() : [];
    for (const el of path) {
      if (!el || !el.classList) continue;
      const c = el.classList;
      if (c.contains('overlay') || c.contains('rail') || c.contains('rail-resize') ||
          c.contains('ctxmenu') || c.contains('confirm') || c.contains('thumb') ||
          c.contains('tapzone') || c.contains('tapzones') ||
          c.contains('deck-hud') || c.contains('advance-hint')) return;
      if (el.tagName === 'A' || el.tagName === 'BUTTON') return;
    }
    nav(1);
  }

  function onSlideChange(e) { update(e.detail.index); }

  function onMessage(ev) {
    const d = ev.data;
    if (d && typeof d.__omelette_presenting === 'boolean') {
      presenting = d.__omelette_presenting;
      if (presenting) {
        metaEl && metaEl.removeAttribute('data-show');
        hint && hint.removeAttribute('data-show');
      }
    }
  }

  function start() {
    if (!build()) { setTimeout(start, 120); return; }
    const d = deck();
    if (d) d.addEventListener('slidechange', onSlideChange);
    window.addEventListener('wheel', onWheel, { passive: true });
    document.addEventListener('click', onClick);
    window.addEventListener('mousemove', showMeta, { passive: true });
    window.addEventListener('message', onMessage);
    const active = slides.findIndex((s) => s.hasAttribute('data-deck-active'));
    update(active >= 0 ? active : 0);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(start, 60));
  } else {
    setTimeout(start, 60);
  }
})();
