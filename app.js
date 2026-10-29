'use strict';

(() => {
  const view = document.getElementById('view');       // particles (WebGL2, 2D fallback)
  const overlay = document.getElementById('overlay'); // zones and cursor
  const octx = overlay.getContext('2d');
  const MONO = getComputedStyle(document.body).fontFamily;
  const TAU = Math.PI * 2;
  const PARTICLE = '#333';  // keep in sync with the fragment shader colour
  const ZONE = '#ff6a00';
  const ZONE_DIM = 'rgba(255,106,0,0.85)';
  const LABEL_BG = 'rgba(51,51,51,0.92)';
  const WALL = '#333';
  const WALL_W = 5;      // wall thickness, CSS px
  const WINDOW_HALF = 60; // half the width of a clicked-open window, CSS px

  // ------------------------------------------------------------------ state

  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  // A page opened in a hidden tab can report a near-zero size at load.
  const area = window.innerWidth * window.innerHeight < 100000
    ? 1280 * 800
    : window.innerWidth * window.innerHeight;

  const params = {
    visc: 0,
    vorticity: 1,
    buoyancy: 12,
    cooling: 0.35,
    stir: 0.5,
    heat: 1,
    pressure: 2.5,
    radius: 50,
    windSpeed: 0,
    windDir: 90,
    gust: 0.3,
    count: Math.round(clamp(area / 70, 8000, 30000) / 1000) * 1000,
    size: 3,
    streak: 0,
    grid: clamp(Math.round(Math.sqrt(area / 28000)), 5, 12),
  };
  const state = { tool: 'stir', paused: false, walls: false };

  let W = 0, H = 0, dpr = 1, cs = params.grid;
  let sim = null;

  // Particles, interleaved as x, y, vx, vy in grid units. This array is
  // uploaded to the GPU as-is. It has spare room, since with open edges the
  // number of particles on screen rises and falls with the flow.
  let count = 0;
  let P = new Float32Array(0);
  const room = (n) => new Float32Array(Math.ceil(n * 1.5) * 4);

  const zones = []; // { type, x, y, r } in CSS pixels
  const windows = []; // { side, a, b }, see the windows section
  let hoverWindow = null;
  const controlRefs = {};
  let layoutPending = false;
  const pointers = new Map();
  let hover = null, hoverZone = null;

  // -------------------------------------------------------------- viscosity

  const viscOf = (s) => (s <= 0 ? 0 : 0.01 * Math.pow(10, s * 0.047));
  // Wind direction: degrees clockwise from up, the way the wind blows.
  const ARROWS = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖'];
  const arrowOf = (deg) => ARROWS[Math.round(((deg % 360) + 360) % 360 / 45) % 8];

  // Current wind, with gusts: smooth pseudo-random wobble of speed and
  // direction from a few incommensurate sines.
  const wind = { speed: 0, angle: 90 };
  function updateWind(t) {
    const g = params.gust;
    const n1 = (Math.sin(t * 0.7) + 0.6 * Math.sin(t * 1.9 + 1.3) + 0.3 * Math.sin(t * 3.7 + 0.4)) / 1.9;
    const n2 = (Math.sin(t * 0.43 + 2.1) + 0.5 * Math.sin(t * 1.3 + 0.7)) / 1.5;
    wind.speed = params.windSpeed * Math.max(0, 1 + 0.7 * g * n1);
    wind.angle = params.windDir + 30 * g * n2;
    const a = (wind.angle * Math.PI) / 180;
    sim.windU = (Math.sin(a) * wind.speed) / cs;
    sim.windV = (-Math.cos(a) * wind.speed) / cs;
  }

  const viscWord =(v) => (v < 0.2 ? 'thin' : v < 3 ? 'water' : v < 40 ? 'oil' : 'honey');
  const num = (v) => (v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : String(Math.round(v)));

  // --------------------------------------------------------------- renderer

  function createGLRenderer(canvas) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false });
    if (!gl) return null;

    // One instanced quad per particle. With streaks off it is a square dot;
    // with streaks on it stretches backwards along the particle's velocity.
    const vsSrc = `#version 300 es
      layout(location = 0) in vec2 corner;   // x: 0..1 along, y: -0.5..0.5 across
      layout(location = 1) in vec4 particle; // x, y, vx, vy in grid units
      uniform vec2 uScale;   // clip units per grid unit
      uniform float uSize;   // dot size in grid units
      uniform float uStreak; // seconds of motion shown as a trail
      void main() {
        vec2 trail = particle.zw * uStreak;
        float len = length(trail);
        vec2 dir = len > 1e-4 ? trail / len : vec2(1.0, 0.0);
        float l = max(len, uSize);
        vec2 p = particle.xy + dir * (0.5 * uSize - corner.x * l) + vec2(-dir.y, dir.x) * corner.y * uSize;
        gl_Position = vec4((p.x - 0.5) * uScale.x - 1.0, 1.0 - (p.y - 0.5) * uScale.y, 0.0, 1.0);
      }`;
    const fsSrc = `#version 300 es
      precision mediump float;
      out vec4 color;
      void main() { color = vec4(0.2, 0.2, 0.2, 1.0); }`;

    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    const uScale = gl.getUniformLocation(prog, 'uScale');
    const uSize = gl.getUniformLocation(prog, 'uSize');
    const uStreak = gl.getUniformLocation(prog, 'uStreak');

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -0.5, 1, -0.5, 0, 0.5, 1, 0.5]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const inst = gl.createBuffer();
    let capacity = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, inst);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(1, 1);

    return {
      kind: 'webgl2',
      draw() {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(1, 1, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        if (!count) return;
        gl.useProgram(prog);
        gl.bindVertexArray(vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, inst);
        if (P.length > capacity) {
          capacity = P.length;
          gl.bufferData(gl.ARRAY_BUFFER, capacity * 4, gl.DYNAMIC_DRAW);
        }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, P, 0, count * 4);
        gl.uniform2f(uScale, (2 * cs) / W, (2 * cs) / H);
        gl.uniform1f(uSize, params.size / cs);
        gl.uniform1f(uStreak, params.streak / 60);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
      },
    };
  }

  // Fallback for browsers without WebGL2.
  function create2DRenderer(canvas) {
    const ctx = canvas.getContext('2d', { alpha: false });
    return {
      kind: '2d',
      draw() {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = ctx.strokeStyle = PARTICLE;
        const s = params.size, h = s / 2, k = (params.streak / 60) * cs;
        ctx.lineWidth = s;
        ctx.beginPath();
        for (let i = 0, end = count * 4; i < end; i += 4) {
          const x = (P[i] - 0.5) * cs, y = (P[i + 1] - 0.5) * cs;
          ctx.rect(x - h, y - h, s, s);
        }
        ctx.fill();
        if (k > 0) {
          ctx.beginPath();
          for (let i = 0, end = count * 4; i < end; i += 4) {
            const x = (P[i] - 0.5) * cs, y = (P[i + 1] - 0.5) * cs;
            ctx.moveTo(x, y);
            ctx.lineTo(x - P[i + 2] * k, y - P[i + 3] * k);
          }
          ctx.stroke();
        }
      },
    };
  }

  let renderer = null;
  try { renderer = createGLRenderer(view); } catch (err) { console.warn(err); }
  if (!renderer) renderer = create2DRenderer(view);

  // ------------------------------------------------------------ setup / grid

  function randomizeParticle(i) {
    P[i] = 0.5 + Math.random() * sim.nx;
    P[i + 1] = 0.5 + Math.random() * sim.ny;
    P[i + 2] = P[i + 3] = 0;
  }

  function setCount(n) {
    const old = P, keep = Math.min(count, n);
    P = room(n);
    P.set(old.subarray(0, keep * 4));
    count = n;
    for (let i = keep * 4; i < n * 4; i += 4) randomizeParticle(i);
  }

  // Lay the particles out as an evenly spaced matrix over the screen. The
  // count is rounded to the nearest full columns x rows.
  function layoutMatrix(target) {
    const spacing = Math.sqrt((W * H) / target);
    const cols = Math.max(1, Math.round(W / spacing));
    const rows = Math.max(1, Math.round(H / spacing));
    count = cols * rows;
    P = room(count);
    const dx = W / cols, dy = H / rows;
    let i = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++, i += 4) {
        P[i] = ((c + 0.5) * dx) / cs + 0.5;
        P[i + 1] = ((r + 0.5) * dy) / cs + 0.5;
      }
    }
    params.count = count;
    const ref = controlRefs.count;
    if (ref) ref.out.textContent = ref.c.fmt(count);
  }

  function build() {
    const hadSim = !!sim, oldW = W, oldH = H;
    const oldNx = hadSim ? sim.nx : 1, oldNy = hadSim ? sim.ny : 1;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth;
    H = window.innerHeight;
    for (const c of [view, overlay]) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    if (oldW && oldH) {
      for (const z of zones) { z.x *= W / oldW; z.y *= H / oldH; }
      for (const wd of windows) {
        const k = horizontal(wd.side) ? W / oldW : H / oldH;
        wd.a *= k;
        wd.b *= k;
      }
    }

    cs = params.grid;
    sim = new FluidSim(Math.max(8, Math.ceil(W / cs)), Math.max(8, Math.ceil(H / cs)));
    sim.walls = state.walls;

    // Lay out the starting matrix once the page has a real size (a tab that
    // is hidden at load can report ~0).
    if (!hadSim || layoutPending) {
      layoutPending = W < 100 || H < 100;
      if (!layoutPending) layoutMatrix(params.count);
      return;
    }
    // Stretch the particles over the new domain.
    const kx = sim.nx / oldNx, ky = sim.ny / oldNy;
    for (let i = 0; i < count * 4; i += 4) {
      P[i] = 0.5 + (P[i] - 0.5) * kx;
      P[i + 1] = 0.5 + (P[i + 1] - 0.5) * ky;
      P[i + 2] = P[i + 3] = 0;
    }
  }

  function resetFluid() {
    sim.reset();
    layoutMatrix(params.count);
  }

  let rebuildTimer = 0;
  const scheduleRebuild = () => {
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => { rebuildTimer = 0; build(); }, 120);
  };

  // --------------------------------------------------------------- particles

  // Particles are never created or destroyed inside the fluid: they only
  // leave by being carried off an open edge, and enter with fluid flowing
  // in from outside.
  function stepParticles(dt) {
    const { u, v, w, nx, ny } = sim;
    const xmax = nx + 0.5, ymax = ny + 0.5;
    const walls = state.walls;

    // Bilinear velocity lookup, written into su / sv.
    let su = 0, sv = 0;
    const sample = (x, y) => {
      if (x < 0.5) x = 0.5; else if (x > xmax) x = xmax;
      if (y < 0.5) y = 0.5; else if (y > ymax) y = ymax;
      const i0 = x | 0, j0 = y | 0;
      const s1 = x - i0, t1 = y - j0, s0 = 1 - s1, t0 = 1 - t1;
      const k = i0 + w * j0;
      su = s0 * (t0 * u[k] + t1 * u[k + w]) + s1 * (t0 * u[k + 1] + t1 * u[k + 1 + w]);
      sv = s0 * (t0 * v[k] + t1 * v[k + w]) + s1 * (t0 * v[k + 1] + t1 * v[k + 1 + w]);
    };

    // Whether a particle that crossed the boundary at (x, y) left through an
    // open window.
    const { openTop, openBottom, openLeft, openRight } = sim;
    const col = (x) => Math.min(nx, Math.max(1, Math.round(x)));
    const row = (y) => Math.min(ny, Math.max(1, Math.round(y)));
    const throughWindow = (x, y) =>
      y < 0.5 ? openTop[col(x)] : y > ymax ? openBottom[col(x)]
        : x < 0.5 ? openLeft[row(y)] : openRight[row(y)];

    const half = 0.5 * dt;
    let i = 0;
    while (i < count * 4) {
      let x = P[i], y = P[i + 1];
      // Midpoint (RK2) integration through the velocity field.
      sample(x, y);
      sample(x + half * su, y + half * sv);
      x += dt * su;
      y += dt * sv;

      if (x < 0.5 || x > xmax || y < 0.5 || y > ymax) {
        if (!walls || throughWindow(x, y)) {
          // Carried off screen: move the last particle into this slot.
          const last = --count * 4;
          P[i] = P[last]; P[i + 1] = P[last + 1]; P[i + 2] = P[last + 2]; P[i + 3] = P[last + 3];
          continue;
        }
        x = x < 0.5 ? 0.5 : x >= xmax ? xmax - 0.001 : x;
        y = y < 0.5 ? 0.5 : y >= ymax ? ymax - 0.001 : y;
      }
      P[i] = x;
      P[i + 1] = y;
      P[i + 2] = su;
      P[i + 3] = sv;
      i += 4;
    }
  }

  // Open edges and windows: fluid flowing in from outside carries particles
  // at the starting density. Each open edge cell spawns, on average, inflow
  // speed x dt x density particles, placed within the strip that just
  // flowed in.
  function seedInflow(dt) {
    const { u, v, w, nx, ny, openTop, openBottom, openLeft, openRight } = sim;
    const walls = state.walls;
    if (walls && !sim.hasWindows) return;
    const density = (params.count * cs * cs) / (W * H); // particles per cell
    const spawn = (speed, place, open) => {
      if (speed <= 0 || (walls && !open)) return;
      for (let e = speed * dt * density; e > 0; e -= 1) {
        if (Math.random() >= e || count * 4 >= P.length) continue;
        const i = count++ * 4;
        place(i, Math.random(), Math.random() * speed * dt);
        P[i + 2] = P[i + 3] = 0;
      }
    };
    for (let c = 1; c <= nx; c++) {
      spawn(v[c + w], (i, a, d) => { P[i] = c - 0.5 + a; P[i + 1] = 0.5 + d; }, openTop[c]);
      spawn(-v[c + w * ny], (i, a, d) => { P[i] = c - 0.5 + a; P[i + 1] = ny + 0.5 - d; }, openBottom[c]);
    }
    for (let r = 1; r <= ny; r++) {
      spawn(u[1 + w * r], (i, a, d) => { P[i] = 0.5 + d; P[i + 1] = r - 0.5 + a; }, openLeft[r]);
      spawn(-u[nx + w * r], (i, a, d) => { P[i] = nx + 0.5 - d; P[i + 1] = r - 0.5 + a; }, openRight[r]);
    }
  }

  // ---------------------------------------------------------------- windows

  // Windows are { side, a, b }: a stretch of a wall, in CSS pixels along it.
  const horizontal = (side) => side === 'top' || side === 'bottom';

  function nearestSide(x, y) {
    const d = { top: y, bottom: H - y, left: x, right: W - x };
    let side = 'top';
    for (const s in d) if (d[s] < d[side]) side = s;
    return { side, along: horizontal(side) ? x : y, dist: d[side], len: horizontal(side) ? W : H };
  }

  function hitWindow(x, y) {
    if (!state.walls) return null;
    const n = nearestSide(x, y);
    if (n.dist > 40) return null;
    return windows.find((wd) => wd.side === n.side && n.along >= wd.a - 8 && n.along <= wd.b + 8) || null;
  }

  function removeWindow(wd) {
    const i = windows.indexOf(wd);
    if (i >= 0) windows.splice(i, 1);
    if (hoverWindow === wd) hoverWindow = null;
  }

  function applyWindows() {
    sim.clearWindows();
    for (const wd of windows) {
      sim.openWindow(wd.side, Math.floor(wd.a / cs) + 1, Math.ceil(wd.b / cs));
    }
  }

  // ------------------------------------------------------------- simulation

  function applyZones(dt) {
    const gridZones = zones.map((z) => ({ type: z.type, x: z.x / cs + 0.5, y: z.y / cs + 0.5, r: z.r / cs }));
    sim.applyZones(gridZones, params.heat, params.pressure, dt);
  }

  function applyStir(dt) {
    const r = params.radius / cs;
    for (const p of pointers.values()) {
      if (p.mode !== 'stir') continue;
      const dx = p.x - p.lx, dy = p.y - p.ly;
      p.lx = p.x; p.ly = p.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 0.01) continue;
      // Drag velocity in cells per second, scaled by the stir strength.
      let vx = (dx / dt / cs) * params.stir, vy = (dy / dt / cs) * params.stir;
      const speed = Math.hypot(vx, vy), max = sim.maxSpeed;
      if (speed > max) { vx *= max / speed; vy *= max / speed; }
      const steps = Math.min(16, Math.max(1, Math.ceil(dist / (params.radius * 0.5))));
      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        const x = p.x - dx * (1 - t), y = p.y - dy * (1 - t);
        sim.splatVelocity(x / cs + 0.5, y / cs + 0.5, r, vx, vy, 0.35 / steps);
      }
    }
  }

  // ---------------------------------------------------------------- overlay

  function drawOverlay(t) {
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    octx.clearRect(0, 0, W, H);
    drawWalls();
    drawZones(t);
    drawWind();
    drawCursor();
  }

  // Compass in the top-right corner: an orange arrow the way the wind
  // blows, and its current speed.
  function drawWind() {
    if (params.windSpeed <= 0) return;
    const cx = W - 48, cy = 48, R = 22;
    const a = (wind.angle * Math.PI) / 180;
    const dx = Math.sin(a), dy = -Math.cos(a);
    octx.globalAlpha = 1;
    octx.fillStyle = 'rgba(255,255,255,0.9)';
    octx.strokeStyle = WALL;
    octx.lineWidth = 1;
    octx.beginPath();
    octx.arc(cx, cy, R, 0, TAU);
    octx.fill();
    octx.stroke();
    octx.strokeStyle = octx.fillStyle = ZONE;
    octx.lineWidth = 2.5;
    const tx = cx + dx * 14, ty = cy + dy * 14;
    octx.beginPath();
    octx.moveTo(cx - dx * 14, cy - dy * 14);
    octx.lineTo(tx, ty);
    octx.stroke();
    octx.beginPath();
    octx.moveTo(tx, ty);
    octx.lineTo(tx - dx * 7 - dy * 5, ty - dy * 7 + dx * 5);
    octx.lineTo(tx - dx * 7 + dy * 5, ty - dy * 7 - dx * 5);
    octx.closePath();
    octx.fill();
    drawLabel(`wind ${Math.round(wind.speed)} px/s`, cx + R - 110, cy + R + 16);
  }

  // Name tag: white text on a dark grey backdrop, kept on screen.
  function drawLabel(text, x, y) {
    octx.font = `600 12px ${MONO}`;
    octx.textBaseline = 'middle';
    const pad = 6, h = 20;
    const w = octx.measureText(text).width + pad * 2;
    x = clamp(x, 8, W - w - 8);
    y = clamp(y, h / 2 + 8, H - h / 2 - 8);
    octx.globalAlpha = 1;
    octx.fillStyle = LABEL_BG;
    octx.fillRect(x, y - h / 2, w, h);
    octx.fillStyle = '#fff';
    octx.fillText(text, x + pad, y + 1);
  }

  // Walls as solid bars along the screen edges, read straight from the
  // solver's masks; windows as orange gaps with jambs at each end.
  function drawWalls() {
    if (!state.walls) return;
    const T = WALL_W;
    octx.fillStyle = WALL;
    const runs = (mask, n, draw) => {
      let start = -1;
      for (let i = 1; i <= n + 1; i++) {
        const closed = i <= n && !mask[i];
        if (closed && start < 0) start = i;
        if (!closed && start >= 0) { draw((start - 1) * cs, (i - 1) * cs); start = -1; }
      }
    };
    runs(sim.openTop, sim.nx, (a, b) => octx.fillRect(a, 0, b - a, T));
    runs(sim.openBottom, sim.nx, (a, b) => octx.fillRect(a, H - T, b - a, T));
    runs(sim.openLeft, sim.ny, (a, b) => octx.fillRect(0, a, T, b - a));
    runs(sim.openRight, sim.ny, (a, b) => octx.fillRect(W - T, a, T, b - a));

    for (const wd of windows) {
      const hot = wd === hoverWindow;
      octx.strokeStyle = octx.fillStyle = hot ? ZONE : ZONE_DIM;
      octx.lineWidth = hot ? 3 : 2;
      const a = (Math.floor(wd.a / cs)) * cs, b = Math.ceil(wd.b / cs) * cs;
      const J = 16; // jamb length, into the room
      octx.beginPath();
      octx.setLineDash([4, 4]);
      if (horizontal(wd.side)) {
        const y = wd.side === 'top' ? T / 2 : H - T / 2;
        octx.moveTo(a, y); octx.lineTo(b, y);
      } else {
        const x = wd.side === 'left' ? T / 2 : W - T / 2;
        octx.moveTo(x, a); octx.lineTo(x, b);
      }
      octx.stroke();
      octx.setLineDash([]);
      if (wd.side === 'top') { octx.fillRect(a - 2, 0, 4, J); octx.fillRect(b - 2, 0, 4, J); }
      if (wd.side === 'bottom') { octx.fillRect(a - 2, H - J, 4, J); octx.fillRect(b - 2, H - J, 4, J); }
      if (wd.side === 'left') { octx.fillRect(0, a - 2, J, 4); octx.fillRect(0, b - 2, J, 4); }
      if (wd.side === 'right') { octx.fillRect(W - J, a - 2, J, 4); octx.fillRect(W - J, b - 2, J, 4); }
      if (hot) {
        const mid = (a + b) / 2;
        if (wd.side === 'top') drawLabel('window', mid - 28, J + 16);
        else if (wd.side === 'bottom') drawLabel('window', mid - 28, H - J - 16);
        else if (wd.side === 'left') drawLabel('window', J + 8, mid);
        else drawLabel('window', W - J - 80, mid);
      }
    }
  }

  function drawZones(t) {
    for (const z of zones) {
      const hot = z === hoverZone && state.tool !== 'stir';
      octx.strokeStyle = octx.fillStyle = hot ? ZONE : ZONE_DIM;
      octx.lineWidth = hot ? 3 : 2;

      const pressureZone = z.type === 'low' || z.type === 'high';
      octx.setLineDash(pressureZone ? [6, 5] : []);
      octx.beginPath();
      octx.arc(z.x, z.y, z.r, 0, TAU);
      octx.stroke();
      octx.setLineDash([]);

      if (pressureZone) {
        // Dots that stream inwards (low) or outwards (high).
        const low = z.type === 'low';
        for (let a = 0; a < 8; a++) {
          const ang = (a / 8) * TAU + Math.PI / 8;
          for (let m = 0; m < 2; m++) {
            const ph = (t * 0.4 + m * 0.5 + a * 0.0625) % 1;
            const rr = z.r * (0.2 + 0.7 * (low ? 1 - ph : ph));
            octx.globalAlpha = Math.sin(ph * Math.PI);
            octx.fillRect(z.x + Math.cos(ang) * rr - 1.5, z.y + Math.sin(ang) * rr - 1.5, 3, 3);
          }
        }
      } else {
        // Chevrons drifting up (heat) or down (cold).
        const up = z.type === 'heat' ? -1 : 1;
        const c = Math.min(4, z.r * 0.12);
        for (let m = -1; m <= 1; m++) {
          const ph = (t * 0.45 + (m + 1) * 0.33) % 1;
          const x = z.x + m * z.r * 0.35;
          const y = z.y - up * z.r * 0.45 * (1 - 2 * ph);
          octx.globalAlpha = Math.sin(ph * Math.PI);
          octx.beginPath();
          octx.moveTo(x - c, y - up * c);
          octx.lineTo(x, y);
          octx.lineTo(x + c, y - up * c);
          octx.stroke();
        }
      }
      drawLabel(z.type, z.x + z.r * 0.707 + 4, z.y - z.r * 0.707 - 10);
    }
  }

  function drawCursor() {
    if (!hover) return;
    for (const p of pointers.values()) if (p.mode === 'move' || p.mode === 'window') return;
    if (state.tool === 'window') {
      // Preview where a click would open a window.
      if (hoverWindow) return;
      const n = nearestSide(hover.x, hover.y);
      const a = clamp(n.along - WINDOW_HALF, 0, n.len), b = clamp(n.along + WINDOW_HALF, 0, n.len);
      octx.strokeStyle = ZONE_DIM;
      octx.lineWidth = 2;
      octx.setLineDash([4, 4]);
      octx.beginPath();
      const T = WALL_W + 6;
      if (n.side === 'top') { octx.moveTo(a, T); octx.lineTo(b, T); }
      if (n.side === 'bottom') { octx.moveTo(a, H - T); octx.lineTo(b, H - T); }
      if (n.side === 'left') { octx.moveTo(T, a); octx.lineTo(T, b); }
      if (n.side === 'right') { octx.moveTo(W - T, a); octx.lineTo(W - T, b); }
      octx.stroke();
      octx.setLineDash([]);
      return;
    }
    if (state.tool === 'erase' || (hoverZone && state.tool !== 'stir')) return;
    octx.strokeStyle = 'rgba(0,0,0,0.3)';
    octx.lineWidth = 1;
    octx.setLineDash([1, 3]);
    octx.beginPath();
    octx.arc(hover.x, hover.y, params.radius, 0, TAU);
    octx.stroke();
    octx.setLineDash([]);
  }

  // ------------------------------------------------------------------- loop

  const statsEl = document.getElementById('stats');
  let last = performance.now(), fps = 60, simMs = 0, statsAt = 0;

  function frame(now) {
    const real = (now - last) / 1000;
    last = now;
    const dt = clamp(real, 1 / 240, 1 / 30);
    fps += (1 / Math.max(real, 1e-3) - fps) * 0.05;

    // Resize events aren't always delivered (e.g. a tab that was hidden at
    // load), so also check the size here.
    if ((window.innerWidth !== W || window.innerHeight !== H) && !rebuildTimer) scheduleRebuild();

    sim.visc = viscOf(params.visc);
    sim.vorticity = params.vorticity;
    sim.buoyancy = params.buoyancy;
    sim.cooling = params.cooling;
    sim.walls = state.walls;
    applyWindows();
    if (!state.paused) updateWind(now / 1000);

    if (state.paused) {
      for (const p of pointers.values()) { p.lx = p.x; p.ly = p.y; }
    } else {
      const t0 = performance.now();
      applyZones(dt);
      applyStir(dt);
      sim.step(dt);
      stepParticles(dt);
      seedInflow(dt);
      simMs += (performance.now() - t0 - simMs) * 0.05;
    }
    renderer.draw();
    drawOverlay(now / 1000);

    if (now - statsAt > 500) {
      statsAt = now;
      statsEl.textContent =
        `${Math.round(fps)} fps · sim ${simMs.toFixed(1)} ms · ${count.toLocaleString('en-US')} particles · ` +
        `${sim.nx}×${sim.ny} grid` + (state.paused ? ' · paused' : '');
    }
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------ interaction

  function hit(x, y) {
    for (let i = zones.length - 1; i >= 0; i--) {
      const z = zones[i];
      if (Math.hypot(x - z.x, y - z.y) <= Math.max(z.r, 12)) return z;
    }
    return null;
  }

  function removeZone(z) {
    const i = zones.indexOf(z);
    if (i >= 0) zones.splice(i, 1);
    if (hoverZone === z) hoverZone = null;
  }

  function updateCursor() {
    let c = 'crosshair';
    if ((hoverZone || hoverWindow) && state.tool === 'erase') c = 'pointer';
    else if (hoverWindow && state.tool === 'window') c = 'pointer';
    else if (hoverZone && state.tool !== 'stir' && state.tool !== 'window') c = 'grab';
    for (const p of pointers.values()) if (p.mode === 'move') c = 'grabbing';
    view.style.cursor = c;
  }

  view.addEventListener('pointerdown', (e) => {
    const x = e.clientX, y = e.clientY;
    const z = hit(x, y);
    const wd = hitWindow(x, y);
    if (e.button === 2) {
      if (z) removeZone(z);
      else if (wd) removeWindow(wd);
      return;
    }
    view.setPointerCapture(e.pointerId);
    const p = { x, y, lx: x, ly: y, mode: 'stir', zone: null, ox: 0, oy: 0 };
    if (state.tool === 'erase') {
      p.mode = 'erase';
      if (z) removeZone(z);
      else if (wd) removeWindow(wd);
    } else if (state.tool === 'window') {
      // Click a window to close it; click or drag elsewhere to open one on
      // the nearest wall.
      if (wd) {
        removeWindow(wd);
        p.mode = 'none';
      } else {
        const n = nearestSide(x, y);
        p.mode = 'window';
        p.anchor = n.along;
        p.win = { side: n.side, a: clamp(n.along - WINDOW_HALF, 0, n.len), b: clamp(n.along + WINDOW_HALF, 0, n.len) };
        windows.push(p.win);
      }
    } else if (state.tool !== 'stir') {
      p.mode = 'move';
      p.zone = z || { type: state.tool, x, y, r: params.radius };
      if (!z) zones.push(p.zone);
      p.ox = p.zone.x - x;
      p.oy = p.zone.y - y;
    }
    pointers.set(e.pointerId, p);
    updateCursor();
  });

  view.addEventListener('pointermove', (e) => {
    const x = e.clientX, y = e.clientY;
    hover = { x, y };
    const p = pointers.get(e.pointerId);
    if (p) {
      p.x = x; p.y = y;
      if (p.mode === 'move') { p.zone.x = x + p.ox; p.zone.y = y + p.oy; }
      if (p.mode === 'erase') {
        const z = hit(x, y), wd = hitWindow(x, y);
        if (z) removeZone(z);
        else if (wd) removeWindow(wd);
      }
      if (p.mode === 'window') {
        // Dragging sizes the window between the press and the pointer.
        const len = horizontal(p.win.side) ? W : H;
        const along = clamp(horizontal(p.win.side) ? x : y, 0, len);
        if (Math.abs(along - p.anchor) > 10) {
          p.win.a = Math.min(p.anchor, along);
          p.win.b = Math.max(p.anchor, along);
        }
      }
    }
    hoverZone = p && p.mode === 'move' ? p.zone : hit(x, y);
    hoverWindow = p && p.mode === 'window' ? p.win : hoverZone ? null : hitWindow(x, y);
    updateCursor();
  });

  const release = (e) => {
    pointers.delete(e.pointerId);
    if (e.pointerType !== 'mouse') hover = null;
    updateCursor();
  };
  view.addEventListener('pointerup', release);
  view.addEventListener('pointercancel', release);
  view.addEventListener('pointerleave', (e) => {
    if (!pointers.has(e.pointerId)) { hover = null; hoverZone = null; hoverWindow = null; }
  });
  view.addEventListener('contextmenu', (e) => e.preventDefault());

  view.addEventListener('wheel', (e) => {
    e.preventDefault();
    const f = Math.exp(-e.deltaY * 0.0015);
    const z = hit(e.clientX, e.clientY);
    if (z) z.r = clamp(z.r * f, 10, 400);
    else setControl('radius', Math.round(clamp(params.radius * f, 10, 200)));
  }, { passive: false });

  // ---------------------------------------------------------------------- UI

  const TOOLS = [
    ['stir', 'stir'], ['low', 'low p'], ['high', 'high p'],
    ['heat', 'heat'], ['cold', 'cold'], ['window', 'window'],
    ['erase', 'erase'],
  ];
  const HINTS = {
    stir: 'drag to push the fluid around.',
    low: 'click to place a low-pressure zone. it draws fluid in and compresses it until the pressure evens out.',
    high: 'click to place a high-pressure zone. fluid expands outwards until the pressure evens out.',
    heat: 'click to place a heat source. warm fluid rises.',
    cold: 'click to place a cold source. cool fluid sinks.',
    window: 'click near a wall to open a window, or drag along it to size one. click a window to close it. turns walls on.',
    erase: 'click or drag across zones and windows to remove them.',
  };

  const toolsEl = document.getElementById('tools');
  const hintEl = document.getElementById('hint');
  const toolButtons = {};

  function setTool(tool) {
    state.tool = tool;
    for (const [k, b] of Object.entries(toolButtons)) b.classList.toggle('on', k === tool);
    let hint = HINTS[tool];
    if (tool === 'window' && !state.walls) setWalls(true);
    if (tool !== 'stir' && tool !== 'erase' && tool !== 'window') {
      hint += ' drag a zone to move it, scroll to resize, right-click to remove.';
    }
    hintEl.textContent = hint;
    updateCursor();
  }

  TOOLS.forEach(([key, label], i) => {
    const b = document.createElement('button');
    b.innerHTML = `<span>${label}</span><kbd>${i + 1}</kbd>`;
    b.addEventListener('click', () => setTool(key));
    toolsEl.appendChild(b);
    toolButtons[key] = b;
  });

  const CONTROLS = [
    { group: 'fluid' },
    { key: 'visc', label: 'viscosity', min: 0, max: 100, step: 1,
      fmt: (s) => { const v = viscOf(s); return v === 0 ? 'inviscid' : `${num(v)} · ${viscWord(v)}`; } },
    { key: 'vorticity', label: 'swirl', min: 0, max: 5, step: 0.1, fmt: (s) => s.toFixed(1) },
    { key: 'buoyancy', label: 'buoyancy', min: 0, max: 60, step: 1, fmt: String },
    { key: 'cooling', label: 'cooling', min: 0, max: 2, step: 0.01, fmt: (s) => s.toFixed(2) },
    { group: 'forces' },
    { key: 'stir', label: 'stir strength', min: 0.05, max: 1, step: 0.05, fmt: (s) => s.toFixed(2) },
    { key: 'heat', label: 'temperature', min: 0.1, max: 2, step: 0.05, fmt: (s) => s.toFixed(2) },
    { key: 'pressure', label: 'pressure', min: 1, max: 4, step: 0.1,
      fmt: (s) => `${s.toFixed(1)}× density` },
    { key: 'radius', label: 'brush radius', min: 10, max: 200, step: 1, fmt: (s) => `${s}px` },
    { group: 'wind' },
    { key: 'windSpeed', label: 'speed', min: 0, max: 200, step: 5, fmt: (s) => (s === 0 ? 'calm' : `${s} px/s`) },
    { key: 'windDir', label: 'direction', min: 0, max: 355, step: 5, fmt: (s) => `${arrowOf(s)} ${s}°` },
    { key: 'gust', label: 'gusts', min: 0, max: 1, step: 0.05, fmt: (s) => (s === 0 ? 'steady' : s.toFixed(2)) },
    { group: 'particles' },
    { key: 'count', label: 'count', min: 1000, max: 150000, step: 1000,
      fmt: (s) => s.toLocaleString('en-US'), apply: setCount },
    { key: 'size', label: 'size', min: 1, max: 8, step: 0.25, fmt: (s) => `${s.toFixed(2)}px` },
    { key: 'streak', label: 'streaks', min: 0, max: 12, step: 0.5, fmt: (s) => (s === 0 ? 'off' : s.toFixed(1)) },
    { group: 'simulation' },
    { key: 'grid', label: 'cell size', min: 4, max: 16, step: 1,
      fmt: (s) => `${s}px${s <= 5 ? ' · slow' : ''}`, apply: scheduleRebuild },
  ];

  const controlsEl = document.getElementById('controls');
  let groupEl = null;

  for (const c of CONTROLS) {
    if (c.group) {
      groupEl = document.createElement('div');
      groupEl.className = 'group';
      groupEl.innerHTML = `<div class="group-label">${c.group}</div>`;
      controlsEl.appendChild(groupEl);
      continue;
    }
    const row = document.createElement('div');
    row.className = 'row';
    const id = `ctl-${c.key}`;
    row.innerHTML =
      `<label for="${id}" title="double-click to reset">${c.label}</label><output></output>` +
      `<input id="${id}" type="range" min="${c.min}" max="${c.max}" step="${c.step}">`;
    const input = row.querySelector('input'), out = row.querySelector('output');
    const initial = params[c.key];
    input.value = initial;
    out.textContent = c.fmt(initial);
    input.addEventListener('input', () => setControl(c.key, +input.value, true));
    row.querySelector('label').addEventListener('dblclick', () => setControl(c.key, initial));
    controlRefs[c.key] = { c, input, out };
    groupEl.appendChild(row);
  }

  function setControl(key, value, fromInput) {
    const { c, input, out } = controlRefs[key];
    params[key] = value;
    if (!fromInput) input.value = value;
    out.textContent = c.fmt(value);
    if (c.apply) c.apply(value);
  }

  const edgeButtons = document.querySelectorAll('#edges button');
  function setWalls(walls) {
    state.walls = walls;
    edgeButtons.forEach((b) => b.classList.toggle('on', (b.dataset.v === 'walls') === walls));
  }
  edgeButtons.forEach((b) => b.addEventListener('click', () => setWalls(b.dataset.v === 'walls')));

  const pauseBtn = document.getElementById('pause');
  function togglePause() {
    state.paused = !state.paused;
    pauseBtn.textContent = state.paused ? 'resume' : 'pause';
    pauseBtn.classList.toggle('on', state.paused);
  }
  pauseBtn.addEventListener('click', togglePause);
  document.getElementById('reset').addEventListener('click', resetFluid);
  function clearAll() {
    zones.length = 0;
    windows.length = 0;
    hoverZone = hoverWindow = null;
  }
  document.getElementById('clear').addEventListener('click', clearAll);

  const toggleInvert = () => document.documentElement.classList.toggle('invert');
  document.getElementById('invert').addEventListener('click', toggleInvert);

  const panel = document.getElementById('panel');
  const panelBtn = document.getElementById('toggle-panel');
  function togglePanel() {
    const collapsed = panel.classList.toggle('collapsed');
    panelBtn.textContent = collapsed ? '+' : '–';
    panelBtn.title = collapsed ? 'Show panel (H)' : 'Hide panel (H)';
  }
  panelBtn.addEventListener('click', togglePanel);

  // Mouse clicks shouldn't leave buttons focused, so the keyboard shortcuts
  // (space in particular) keep working.
  panel.addEventListener('mousedown', (e) => { if (e.target.closest('button')) e.preventDefault(); });

  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const focused = document.activeElement;
    if (focused && focused.tagName === 'BUTTON' && (e.key === ' ' || e.key === 'Enter')) return;
    const k = e.key.toLowerCase();
    if (k >= '1' && k <= String(TOOLS.length)) setTool(TOOLS[+k - 1][0]);
    else if (k === ' ') { e.preventDefault(); togglePause(); }
    else if (k === 'r') resetFluid();
    else if (k === 'c') clearAll();
    else if (k === 'w') setWalls(!state.walls);
    else if (k === 'i') toggleInvert();
    else if (k === 'h') togglePanel();
    else if (k === '[') setControl('radius', Math.max(10, params.radius - 5));
    else if (k === ']') setControl('radius', Math.min(200, params.radius + 5));
  });

  window.addEventListener('resize', scheduleRebuild);

  // -------------------------------------------------------------------- go

  build();
  setTool('stir');
  requestAnimationFrame(frame);
})();
