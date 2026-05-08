// Particle/data-flow background for hero slides
function initParticles(canvas, opts = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth;
  const H = canvas.height = canvas.clientHeight;
  const N = opts.count || 80;
  const color = opts.color || 'rgba(255,255,255,0.55)';
  const lineColor = opts.lineColor || 'rgba(255,255,255,0.08)';
  const pts = Array.from({length: N}, () => ({
    x: Math.random() * W, y: Math.random() * H,
    vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
    r: 1 + Math.random() * 1.5
  }));
  let raf;
  function tick() {
    ctx.clearRect(0, 0, W, H);
    // Glide path lines (subtle)
    ctx.strokeStyle = 'rgba(120,200,160,0.10)';
    ctx.lineWidth = 1;
    for (let k = 0; k < 4; k++) {
      ctx.beginPath();
      const yBase = H * (0.55 + k * 0.07);
      ctx.moveTo(0, yBase + Math.sin(Date.now() * 0.0003 + k) * 30);
      for (let x = 0; x < W; x += 30) {
        const y = yBase - x * 0.12 + Math.sin(x * 0.005 + Date.now() * 0.0005 + k) * 18;
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    // Particles + connections
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    }
    ctx.strokeStyle = lineColor;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 22000) {
          ctx.globalAlpha = 1 - d2 / 22000;
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    for (const p of pts) {
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
    }
    raf = requestAnimationFrame(tick);
  }
  tick();
  return () => cancelAnimationFrame(raf);
}

// MDP loop animation: 4 nodes, traveling pulse
function initMDP(svg) {
  const cx = 960, cy = 350;
  const r = 240;
  const nodes = [
    { id: 'state',  label: 'STATE',  sub: 'wealth · time · gap · regime', angle: -Math.PI / 2 },
    { id: 'action', label: 'ACTION', sub: 'rebalance allocation',         angle: 0 },
    { id: 'market', label: 'MARKET', sub: 'returns + contributions',      angle: Math.PI / 2 },
    { id: 'reward', label: 'REWARD', sub: 'goal progress − risk',         angle: Math.PI }
  ];
  nodes.forEach(n => { n.x = cx + Math.cos(n.angle) * r; n.y = cy + Math.sin(n.angle) * r; });

  let html = '';
  // arcs between nodes (clockwise)
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i], b = nodes[(i + 1) % nodes.length];
    html += `<path id="arc-${i}" d="M ${a.x} ${a.y} A ${r} ${r} 0 0 1 ${b.x} ${b.y}" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>`;
  }
  // Center label
  html += `<text x="${cx}" y="${cy - 14}" text-anchor="middle" fill="rgba(255,255,255,0.5)" font-family="Geist Mono, monospace" font-size="14" letter-spacing="3">RL AGENT</text>`;
  html += `<text x="${cx}" y="${cy + 22}" text-anchor="middle" fill="#fafaf7" font-family="Geist, sans-serif" font-size="34" font-weight="500" letter-spacing="-1">learns π(s)</text>`;
  // Nodes
  nodes.forEach((n, i) => {
    html += `
      <g>
        <circle cx="${n.x}" cy="${n.y}" r="80" fill="#0a0b0d" stroke="rgba(255,255,255,0.25)" stroke-width="1.5"/>
        <text x="${n.x}" y="${n.y - 6}" text-anchor="middle" fill="#fafaf7" font-family="Geist, sans-serif" font-size="22" font-weight="500" letter-spacing="-0.5">${n.label}</text>
        <text x="${n.x}" y="${n.y + 18}" text-anchor="middle" fill="rgba(255,255,255,0.55)" font-family="Geist Mono, monospace" font-size="11" letter-spacing="1.5">${n.sub.toUpperCase()}</text>
      </g>`;
  });
  // Pulse
  html += `<circle r="8" fill="oklch(0.66 0.16 152)">
    <animateMotion dur="6s" repeatCount="indefinite">
      <mpath href="#arc-0"/>
    </animateMotion>
  </circle>`;
  // 4 dots traveling, staggered
  for (let i = 0; i < 4; i++) {
    html += `<circle r="6" fill="oklch(0.66 0.16 152)" opacity="0.9">
      <animateMotion dur="6s" begin="${i * 1.5}s" repeatCount="indefinite" rotate="auto">
        <mpath href="#arc-${i}"/>
      </animateMotion>
    </circle>`;
  }
  svg.innerHTML = html;
}

// Regime mini-vizes (sparklines that animate)
function initRegimeViz(container, kind) {
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', '0 0 300 100');
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  svg.style.overflow = 'visible';

  const colors = {
    bull: '#3aa974', bear: '#d05a3e', high: '#d4a01e', stab: '#5c84d4'
  };
  const W = 300, H = 100;
  function pathFor(kind, phase) {
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const t = i / 60;
      let y;
      if (kind === 'bull') y = H * (0.85 - t * 0.55) + Math.sin(t * 12 + phase) * 4;
      else if (kind === 'bear') y = H * (0.25 + t * 0.5) + Math.sin(t * 8 + phase) * 5;
      else if (kind === 'high') y = H * 0.5 + Math.sin(t * 22 + phase) * 28 + Math.sin(t * 6 + phase) * 8;
      else y = H * 0.5 + Math.sin(t * 4 + phase) * 6;
      pts.push(`${(t * W).toFixed(1)},${y.toFixed(1)}`);
    }
    return 'M ' + pts.join(' L ');
  }
  const path = document.createElementNS(svgNS, 'path');
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', colors[kind]);
  path.setAttribute('stroke-width', '2');
  path.setAttribute('stroke-linecap', 'round');
  svg.appendChild(path);

  // baseline
  const bl = document.createElementNS(svgNS, 'line');
  bl.setAttribute('x1', 0); bl.setAttribute('x2', W);
  bl.setAttribute('y1', H * 0.5); bl.setAttribute('y2', H * 0.5);
  bl.setAttribute('stroke', 'rgba(255,255,255,0.08)');
  bl.setAttribute('stroke-dasharray', '2 4');
  svg.insertBefore(bl, path);

  container.appendChild(svg);
  let phase = 0;
  function tick() {
    phase += 0.04;
    path.setAttribute('d', pathFor(kind, phase));
    requestAnimationFrame(tick);
  }
  tick();
}

// Race chart: RL vs baselines reaching the goal
function initRaceChart(svg) {
  const W = 1500, H = 600;
  const padL = 80, padR = 40, padT = 40, padB = 80;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const target = 1.0; // normalized goal
  const years = 30;

  const series = [
    { name: 'Buy & Hold',   color: '#8a8a85', vol: 0.18, drift: 0.05, wobble: 0.6 },
    { name: '60/40 fixed',  color: '#a78bfa', vol: 0.13, drift: 0.058, wobble: 0.4 },
    { name: 'Glide path',   color: '#5c84d4', vol: 0.10, drift: 0.062, wobble: 0.3 },
    { name: 'G-Learner',    color: '#d4a01e', vol: 0.085, drift: 0.069, wobble: 0.2 },
    { name: 'Regime-aware', color: 'oklch(0.66 0.16 152)', vol: 0.06, drift: 0.075, wobble: 0.15, hero: true }
  ];

  // Pre-generate paths (deterministic-ish via seeded)
  function seeded(seed) { return () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; }; }
  series.forEach((s, idx) => {
    const rng = seeded(idx * 1000 + 7);
    let v = 0.0;
    const path = [];
    for (let y = 0; y <= years; y++) {
      const t = y / years;
      v = t * (target * (0.85 + idx * 0.06)) + (rng() - 0.5) * s.vol * 0.6 * (1 - t * 0.5);
      // small dips for non-hero in middle (simulating regime crash)
      if (!s.hero && y > 12 && y < 18) v -= 0.08;
      if (s.hero && y > 12 && y < 18) v -= 0.025; // reacts smaller
      path.push(v);
    }
    s.path = path;
  });

  function x(y) { return padL + (y / years) * innerW; }
  function yy(v) { return padT + innerH - Math.max(0, Math.min(1.15, v)) * innerH * 0.85; }

  let html = '';
  // Goal target band
  html += `<rect x="${padL}" y="${yy(target)}" width="${innerW}" height="2" fill="oklch(0.66 0.16 152)"/>`;
  html += `<text x="${padL + innerW - 8}" y="${yy(target) - 14}" text-anchor="end" fill="oklch(0.66 0.16 152)" font-family="Geist Mono, monospace" font-size="14" letter-spacing="2">GOAL · TARGET WEALTH</text>`;

  // Y axis
  for (const v of [0, 0.5, 1.0]) {
    html += `<line x1="${padL}" x2="${padL + innerW}" y1="${yy(v)}" y2="${yy(v)}" stroke="rgba(255,255,255,0.06)"/>`;
    html += `<text x="${padL - 14}" y="${yy(v) + 5}" text-anchor="end" fill="#8a8a85" font-family="Geist Mono, monospace" font-size="12">${(v*100).toFixed(0)}%</text>`;
  }
  // X axis
  for (let yr = 0; yr <= years; yr += 5) {
    html += `<line x1="${x(yr)}" x2="${x(yr)}" y1="${padT + innerH}" y2="${padT + innerH + 6}" stroke="rgba(255,255,255,0.2)"/>`;
    html += `<text x="${x(yr)}" y="${padT + innerH + 28}" text-anchor="middle" fill="#8a8a85" font-family="Geist Mono, monospace" font-size="12">y${yr}</text>`;
  }
  // Series paths (drawn with animation)
  series.forEach((s, idx) => {
    const d = s.path.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${yy(v).toFixed(1)}`).join(' ');
    const len = innerW * 1.4;
    html += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.hero ? 4 : 2}" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="${len}" stroke-dashoffset="${len}" opacity="${s.hero ? 1 : 0.65}">
      <animate attributeName="stroke-dashoffset" from="${len}" to="0" dur="${s.hero ? 4 : 3.5}s" begin="${0.2 + idx * 0.15}s" fill="freeze"/>
    </path>`;
    // End label
    const lastY = s.path[s.path.length - 1];
    html += `<text x="${x(years) + 12}" y="${yy(lastY) + 4}" fill="${s.color}" font-family="Geist Mono, monospace" font-size="12" letter-spacing="1" opacity="0">
      ${s.name.toUpperCase()}
      <animate attributeName="opacity" from="0" to="1" begin="${4 + idx * 0.1}s" dur="0.6s" fill="freeze"/>
    </text>`;
  });

  svg.innerHTML = html;
  return series;
}

// GPS card path animation
function initGPS(svg) {
  const W = 600, H = 600;
  let html = '';
  // grid
  for (let i = 0; i <= 12; i++) {
    html += `<line x1="${i * 50}" x2="${i * 50}" y1="0" y2="${H}" stroke="rgba(255,255,255,0.04)"/>`;
    html += `<line y1="${i * 50}" y2="${i * 50}" x1="0" x2="${W}" stroke="rgba(255,255,255,0.04)"/>`;
  }
  // start
  html += `<circle cx="80" cy="500" r="14" fill="none" stroke="#fafaf7" stroke-width="2"/>`;
  html += `<circle cx="80" cy="500" r="4" fill="#fafaf7"/>`;
  html += `<text x="80" y="540" text-anchor="middle" fill="#fafaf7" font-family="Geist Mono, monospace" font-size="13" letter-spacing="2">START</text>`;
  // target
  html += `<circle cx="500" cy="120" r="22" fill="none" stroke="oklch(0.66 0.16 152)" stroke-width="2"/>`;
  html += `<circle cx="500" cy="120" r="14" fill="none" stroke="oklch(0.66 0.16 152)" stroke-width="1.5"/>`;
  html += `<circle cx="500" cy="120" r="6" fill="oklch(0.66 0.16 152)"/>`;
  html += `<text x="500" y="80" text-anchor="middle" fill="oklch(0.66 0.16 152)" font-family="Geist Mono, monospace" font-size="13" letter-spacing="2">GOAL</text>`;
  // path - safe (curving, deliberate)
  const safe = "M 80 500 Q 200 460, 240 380 T 360 280 T 500 120";
  html += `<path d="${safe}" fill="none" stroke="oklch(0.66 0.16 152)" stroke-width="3" stroke-linecap="round" stroke-dasharray="6 8" opacity="0.9"/>`;
  // path - reckless (zigzag through danger)
  const wild = "M 80 500 L 180 460 L 130 380 L 280 360 L 200 270 L 360 240 L 290 170 L 500 120";
  html += `<path d="${wild}" fill="none" stroke="rgba(220,90,60,0.5)" stroke-width="1.5" stroke-linecap="round"/>`;
  // moving dot on safe
  html += `<circle r="8" fill="oklch(0.66 0.16 152)">
    <animateMotion dur="5s" repeatCount="indefinite" path="${safe}"/>
  </circle>`;
  svg.innerHTML = html;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
}

// Bot side maps (slide 7) — fast/wild vs deliberate/safe
function initBotMap(svg, kind) {
  const W = 700, H = 320;
  let html = '';
  // grid
  for (let i = 0; i <= 14; i++) {
    html += `<line x1="${i * 50}" x2="${i * 50}" y1="0" y2="${H}" stroke="rgba(10,11,13,0.06)"/>`;
  }
  for (let i = 0; i <= 7; i++) {
    html += `<line y1="${i * 50}" y2="${i * 50}" x1="0" x2="${W}" stroke="rgba(10,11,13,0.06)"/>`;
  }
  html += `<circle cx="60" cy="260" r="10" fill="#0a0b0d"/>`;
  html += `<text x="60" y="295" text-anchor="middle" fill="#6b6b66" font-family="Geist Mono, monospace" font-size="12" letter-spacing="2">START</text>`;
  if (kind === 'bot') {
    const path = "M 60 260 L 160 100 L 240 220 L 320 80 L 420 240 L 520 60 L 600 220 L 660 90";
    html += `<path d="${path}" fill="none" stroke="#c8553d" stroke-width="2.5" stroke-linecap="round"/>`;
    html += `<circle cx="660" cy="90" r="10" fill="none" stroke="#c8553d" stroke-width="2"/>`;
    html += `<text x="660" y="60" text-anchor="middle" fill="#c8553d" font-family="Geist Mono, monospace" font-size="12" letter-spacing="2">??</text>`;
    html += `<circle r="7" fill="#c8553d">
      <animateMotion dur="3.5s" repeatCount="indefinite" path="${path}"/>
    </circle>`;
  } else {
    const path = "M 60 260 Q 200 240, 280 200 T 460 130 T 640 60";
    html += `<path d="${path}" fill="none" stroke="oklch(0.55 0.13 152)" stroke-width="3" stroke-linecap="round" stroke-dasharray="8 10"/>`;
    html += `<circle cx="640" cy="60" r="14" fill="none" stroke="oklch(0.55 0.13 152)" stroke-width="2"/>`;
    html += `<circle cx="640" cy="60" r="6" fill="oklch(0.55 0.13 152)"/>`;
    html += `<text x="640" y="36" text-anchor="middle" fill="oklch(0.45 0.13 152)" font-family="Geist Mono, monospace" font-size="12" letter-spacing="2">GOAL</text>`;
    html += `<circle r="7" fill="oklch(0.55 0.13 152)">
      <animateMotion dur="6s" repeatCount="indefinite" path="${path}"/>
    </circle>`;
  }
  svg.innerHTML = html;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
}

window.DeckAnims = { initParticles, initMDP, initRegimeViz, initRaceChart, initGPS, initBotMap };
