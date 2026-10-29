'use strict';

/*
 * Stable-fluids solver (Stam, 1999) on a staggered (MAC) grid with one ghost
 * cell on every side. Grid units throughout: cell (i, j) has its centre at
 * (i, j), velocities are in cells per second.
 *
 * Pressure and density live at cell centres. Velocities live on the faces
 * between cells: u[i + w * j] is the flow through the face at (i + 0.5, j),
 * between cells i and i + 1, and v[i + w * j] through the face at
 * (i, j + 0.5). So u[w * j] and u[nx + w * j] are the left and right edges
 * of the domain, v[i] and v[i + w * ny] the top and bottom. Row 0 and row
 * ny + 1 of u, and column 0 and column nx + 1 of v, are ghosts that hold the
 * velocity along the edges.
 *
 * With velocities on faces, the divergence of a cell is exactly what flows
 * through its four faces, and the pressure solve is converged every step
 * (multigrid-preconditioned conjugate gradients), so the flow the particles
 * ride on carries no more fluid into a cell than out of it.
 *
 * Extras on top of the classic solver:
 *  - fans: a disc of fluid driven towards a set velocity, the way a fan
 *    keeps pushing air through itself. Everything is seen from above, so
 *    there is no gravity and nothing rises;
 *  - vorticity confinement, to keep small swirls from being smeared away;
 *  - weak compressibility: a density field rho (1 = at rest) is carried
 *    with the flow and updated from the velocity divergence. Wherever the
 *    fluid is squeezed (rho > 1) the projection lets it expand again, and
 *    stretched fluid (rho < 1) contracts, at a rate set by the stiffness;
 *  - pressure zones: a low-pressure zone draws fluid in and compresses it,
 *    a high-pressure zone pushes fluid out and lets it expand. The flow
 *    stops once the density inside balances the zone (rho = 1 +/- the
 *    zone's strength), and the fluid relaxes back once the zone is gone.
 *    No fluid is created or destroyed;
 *  - two edge modes: "open" edges sit at ambient pressure, so fluid can flow
 *    in and out of the screen; "walls" closes the box. With walls, single
 *    boundary cells can be opened ("windows"), which then behave like open
 *    edges;
 *  - an outside wind that enters through the open boundary cells facing it,
 *    drags air along past openings, and makes a room with only windward
 *    openings breathe in and out through them.
 */
const WALL = 0, OUTLET = 1, INLET = 2;

class FluidSim {
  constructor(nx, ny) {
    this.nx = nx;
    this.ny = ny;
    this.w = nx + 2;
    const n = this.w * (ny + 2);
    const field = () => new Float32Array(n);
    this.u = field(); this.v = field();
    this.u0 = field(); this.v0 = field();
    this.p = field(); this.div = field(); this.curl = field();
    this.rho = field(); this.rho0 = field();
    this.rho.fill(1);
    this.source = field(); // divergence the zones ask for, per second

    // Window masks, one flag per boundary cell (indexed like the grid).
    this.openTop = new Uint8Array(nx + 2);
    this.openBottom = new Uint8Array(nx + 2);
    this.openLeft = new Uint8Array(ny + 2);
    this.openRight = new Uint8Array(ny + 2);
    this.hasWindows = false;

    // Outside wind, cells per second. It blows in through open boundary
    // cells that face into it.
    this.windU = 0;
    this.windV = 0;
    this.time = 0;
    this.shear = 4; // how strongly wind drags air inside openings, 1/s

    // Boundary cell kinds (WALL / INLET / OUTLET), see updateBoundaryKinds.
    this.kindTop = new Uint8Array(nx + 2);
    this.kindBottom = new Uint8Array(nx + 2);
    this.kindLeft = new Uint8Array(ny + 2);
    this.kindRight = new Uint8Array(ny + 2);

    this.walls = false;
    this.stiffness = 1.5; // how quickly squeezed / stretched fluid relaxes, 1/s
    this.visc = 0;
    this.vorticity = 1;
    this.maxSpeed = 50;
    this.viscIters = 12;
    this.pressure = new PressureSolver(nx, ny);
    this.su = 0; this.sv = 0; // result of sample()
    this.updateBoundaryKinds(0);
  }

  reset() {
    for (const f of [this.u, this.v, this.u0, this.v0, this.p, this.div, this.curl]) f.fill(0);
    this.rho.fill(1);
  }

  // ---------------------------------------------------------------- sampling

  // Bilinear lookups at a point (x, y) inside the domain, one per kind of
  // grid location: faces of u, faces of v, cell centres.
  sampleU(f, x, y) {
    const { nx, w } = this;
    x -= 0.5;
    let i0 = x | 0;
    if (i0 > nx - 1) i0 = nx - 1;
    const j0 = y | 0;
    const s1 = x - i0, t1 = y - j0, s0 = 1 - s1, t0 = 1 - t1;
    const k = i0 + w * j0;
    return s0 * (t0 * f[k] + t1 * f[k + w]) + s1 * (t0 * f[k + 1] + t1 * f[k + 1 + w]);
  }

  sampleV(f, x, y) {
    const { ny, w } = this;
    y -= 0.5;
    let j0 = y | 0;
    if (j0 > ny - 1) j0 = ny - 1;
    const i0 = x | 0;
    const s1 = x - i0, t1 = y - j0, s0 = 1 - s1, t0 = 1 - t1;
    const k = i0 + w * j0;
    return s0 * (t0 * f[k] + t1 * f[k + w]) + s1 * (t0 * f[k + 1] + t1 * f[k + 1 + w]);
  }

  sampleC(f, x, y) {
    const w = this.w;
    const i0 = x | 0, j0 = y | 0;
    const s1 = x - i0, t1 = y - j0, s0 = 1 - s1, t0 = 1 - t1;
    const k = i0 + w * j0;
    return s0 * (t0 * f[k] + t1 * f[k + w]) + s1 * (t0 * f[k + 1] + t1 * f[k + 1 + w]);
  }

  // Velocity at (x, y), clamped into the domain, written into su / sv.
  sample(x, y) {
    const xmax = this.nx + 0.5, ymax = this.ny + 0.5;
    if (x < 0.5) x = 0.5; else if (x > xmax) x = xmax;
    if (y < 0.5) y = 0.5; else if (y > ymax) y = ymax;
    this.su = this.sampleU(this.u, x, y);
    this.sv = this.sampleV(this.v, x, y);
  }

  // How fast fluid flows into the domain through boundary cell i of a side.
  inflow(side, i) {
    const { nx, ny, w, u, v } = this;
    if (side === 'top') return v[i];
    if (side === 'bottom') return -v[i + w * ny];
    if (side === 'left') return u[w * i];
    return -u[nx + w * i];
  }

  // ---------------------------------------------------------------- boundaries

  corners(x) {
    const { nx, ny, w } = this;
    const last = w * (ny + 1);
    x[0] = 0.5 * (x[1] + x[w]);
    x[nx + 1] = 0.5 * (x[nx] + x[nx + 1 + w]);
    x[last] = 0.5 * (x[last + 1] + x[last - w]);
    x[last + nx + 1] = 0.5 * (x[last + nx] + x[last - w + nx + 1]);
  }

  // Close all windows.
  clearWindows() {
    this.openTop.fill(0);
    this.openBottom.fill(0);
    this.openLeft.fill(0);
    this.openRight.fill(0);
    this.hasWindows = false;
  }

  // Open boundary cells i0..i1 on one side ('top' | 'bottom' | 'left' | 'right').
  openWindow(side, i0, i1) {
    const mask = side === 'top' ? this.openTop : side === 'bottom' ? this.openBottom
      : side === 'left' ? this.openLeft : this.openRight;
    const n = mask.length - 2;
    for (let i = Math.max(1, i0); i <= Math.min(n, i1); i++) mask[i] = 1;
    this.hasWindows = true;
  }

  // Works out the kind of every boundary cell from the walls, windows and
  // wind: WALL, INLET (open and facing into the wind, which blows in at its
  // own velocity) or OUTLET (open, at ambient pressure).
  updateBoundaryKinds(dt) {
    this.time += dt;
    const { walls, windU: wu, windV: wv } = this;
    const windy = wu !== 0 || wv !== 0;
    const sides = [
      // kinds, window mask, wind component into the room, wind along the side
      [this.kindTop, this.openTop, wv, wu],
      [this.kindBottom, this.openBottom, -wv, wu],
      [this.kindLeft, this.openLeft, wu, wv],
      [this.kindRight, this.openRight, -wu, wv],
    ];
    let outlets = false;
    for (const [kind, open, inward] of sides) {
      for (let i = 1; i < kind.length - 1; i++) {
        kind[i] = walls && !open[i] ? WALL : windy && inward > 0 ? INLET : OUTLET;
        if (kind[i] === OUTLET) outlets = true;
      }
    }
    if (outlets || !windy) return;

    // Every opening faces into the wind, e.g. a single window. A real room
    // then breathes through each opening: air comes in through the upwind
    // half and leaves through the downwind half. With head-on wind the
    // halves slowly take turns.
    const turn = Math.sin(this.time * 0.35) > 0;
    for (const [kind, , inward, along] of sides) {
      const outFirst = Math.abs(along) > 0.2 * inward ? along < 0 : turn;
      let i = 1;
      while (i < kind.length - 1) {
        if (kind[i] !== INLET) { i++; continue; }
        let j = i;
        while (j + 1 < kind.length - 1 && kind[j + 1] === INLET) j++;
        const split = i + ((j - i + 1) >> 1);
        for (let c = i; c <= j; c++) {
          if (i === j || (c < split) === outFirst) kind[c] = OUTLET;
        }
        i = j + 1;
      }
    }
  }

  // Flow through the edges of the domain. Walls let nothing through, inlets
  // hold the wind velocity. At outlets this is only a first guess for the
  // projection, which settles the real outflow: outflow from the next face
  // in passes through, inflow isn't copied (copying backflow is a classic
  // runaway), and air can still be drawn in there by the pressure.
  normalBoundary() {
    const { nx, ny, w, u, v, kindTop, kindBottom, kindLeft, kindRight, windU: wu, windV: wv } = this;
    for (let i = 1; i <= nx; i++) {
      const t = i, b = i + w * ny;
      const kt = kindTop[i], kb = kindBottom[i];
      v[t] = kt === WALL ? 0 : kt === INLET ? wv : Math.min(v[t + w], 0);
      v[b] = kb === WALL ? 0 : kb === INLET ? wv : Math.max(v[b - w], 0);
    }
    for (let j = 1; j <= ny; j++) {
      const l = w * j, r = l + nx;
      const kl = kindLeft[j], kr = kindRight[j];
      u[l] = kl === WALL ? 0 : kl === INLET ? wu : Math.min(u[l + 1], 0);
      u[r] = kr === WALL ? 0 : kr === INLET ? wu : Math.max(u[r - 1], 0);
    }
  }

  // Flow along the edges, in the ghost row / column just outside. Walls
  // slip freely, inlets hold the wind, and outside air streaming past an
  // outlet drags its tangential component along with the wind.
  tangentialBoundary() {
    const { nx, ny, w, u, v, kindTop, kindBottom, kindLeft, kindRight, windU: wu, windV: wv } = this;
    const windy = wu !== 0 || wv !== 0;
    const last = w * (ny + 1);
    for (let i = 0; i <= nx; i++) {
      const c = i < 1 ? 1 : i;
      const kt = kindTop[c], kb = kindBottom[c];
      u[i] = kt === INLET || (kt === OUTLET && windy) ? wu : u[i + w];
      u[last + i] = kb === INLET || (kb === OUTLET && windy) ? wu : u[last + i - w];
    }
    for (let j = 0; j <= ny; j++) {
      const c = j < 1 ? 1 : j, l = w * j, r = l + nx + 1;
      const kl = kindLeft[c], kr = kindRight[c];
      v[l] = kl === INLET || (kl === OUTLET && windy) ? wv : v[l + 1];
      v[r] = kr === INLET || (kr === OUTLET && windy) ? wv : v[r - 1];
    }
  }

  velocityBoundary() {
    this.normalBoundary();
    this.tangentialBoundary();
  }

  // Turbulent shear at openings: the outside wind streaming past an opening
  // drags the air just inside it along. The first two rows of faces inside
  // each outlet relax their along-the-wall velocity towards the wind's.
  applyWindShear(dt) {
    const { nx, ny, w, u, v, kindTop, kindBottom, kindLeft, kindRight, windU: wu, windV: wv } = this;
    if (wu === 0 && wv === 0) return;
    const f1 = 1 - Math.exp(-this.shear * dt), f2 = 0.5 * f1;
    // A face along an edge is at an outlet if either cell beside it is.
    const out = (kind, i) => kind[i] === OUTLET || kind[i + 1] === OUTLET;
    for (let i = 1; i < nx; i++) {
      if (out(kindTop, i)) {
        const k = i + w;
        u[k] += (wu - u[k]) * f1;
        u[k + w] += (wu - u[k + w]) * f2;
      }
      if (out(kindBottom, i)) {
        const k = i + w * ny;
        u[k] += (wu - u[k]) * f1;
        u[k - w] += (wu - u[k - w]) * f2;
      }
    }
    for (let j = 1; j < ny; j++) {
      if (out(kindLeft, j)) {
        const k = 1 + w * j;
        v[k] += (wv - v[k]) * f1;
        v[k + 1] += (wv - v[k + 1]) * f2;
      }
      if (out(kindRight, j)) {
        const k = nx + w * j;
        v[k] += (wv - v[k]) * f1;
        v[k - 1] += (wv - v[k - 1]) * f2;
      }
    }
  }

  // Open boundary cells hold the ambient value; walls are zero-gradient
  // (insulating, no flux).
  scalarBoundary(x, ambient = 0) {
    const { nx, ny, w, kindTop, kindBottom, kindLeft, kindRight } = this;
    const last = w * (ny + 1);
    for (let i = 1; i <= nx; i++) {
      x[i] = kindTop[i] !== WALL ? ambient : x[i + w];
      x[last + i] = kindBottom[i] !== WALL ? ambient : x[last + i - w];
    }
    for (let j = 1; j <= ny; j++) {
      const r = w * j;
      x[r] = kindLeft[j] !== WALL ? ambient : x[r + 1];
      x[r + nx + 1] = kindRight[j] !== WALL ? ambient : x[r + nx];
    }
    this.corners(x);
  }

  // -------------------------------------------------------------- user input

  // zones: [{ type: 'low' | 'high' | 'fan', x, y, r, angle }] in grid units,
  // angle (fans only) in radians from +x towards +y. squeeze is how many
  // times denser (low) or thinner (high) the fluid in a pressure zone
  // settles at; fanSpeed is in cells per second.
  applyZones(zones, fanSpeed, squeeze, dt) {
    const { nx, ny, w, source } = this;
    source.fill(0);
    const rate = Math.min(1, 6 * dt);
    // At rest, stiffness * (rho - 1) + source = 0 inside the zone.
    const qLow = -this.stiffness * (squeeze - 1);
    const qHigh = -this.stiffness * (1 / squeeze - 1);
    for (const z of zones) {
      const r = Math.max(z.r, 0.75), r2 = r * r;
      if (z.type === 'fan') {
        const speed = Math.min(fanSpeed, this.maxSpeed);
        this.splatVelocity(z.x, z.y, r, Math.cos(z.angle) * speed, Math.sin(z.angle) * speed, rate);
        continue;
      }
      const i0 = Math.max(1, Math.floor(z.x - r)), i1 = Math.min(nx, Math.ceil(z.x + r));
      const j0 = Math.max(1, Math.floor(z.y - r)), j1 = Math.min(ny, Math.ceil(z.y + r));
      const zq = z.type === 'low' ? qLow : qHigh;
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const dx = i - z.x, dy = j - z.y, d2 = dx * dx + dy * dy;
          if (d2 > r2) continue;
          source[i + w * j] += zq;
        }
      }
    }
  }

  // Nudges the velocity under a soft circular brush towards (vx, vy). Each
  // face is weighted by its own distance from the centre.
  splatVelocity(cx, cy, r, vx, vy, strength) {
    const { nx, ny, w, u, v } = this;
    const r2 = r * r;
    const brush = (f, target, ox, oy, imin, imax, jmin, jmax) => {
      const i0 = Math.max(imin, Math.floor(cx - ox - r)), i1 = Math.min(imax, Math.ceil(cx - ox + r));
      const j0 = Math.max(jmin, Math.floor(cy - oy - r)), j1 = Math.min(jmax, Math.ceil(cy - oy + r));
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const dx = i + ox - cx, dy = j + oy - cy, d2 = dx * dx + dy * dy;
          if (d2 > r2) continue;
          const k = i + w * j;
          f[k] += (target - f[k]) * strength * Math.exp(-3 * d2 / r2);
        }
      }
    };
    brush(u, vx, 0.5, 0, 1, nx - 1, 1, ny);
    brush(v, vy, 0, 0.5, 1, nx, 1, ny - 1);
  }

  // ------------------------------------------------------------------ solver

  addForces(dt) {
    const { nx, ny, w, u, v, u0: fx, v0: fy, curl } = this;

    // Vorticity confinement: push along N x omega, where N points towards
    // stronger swirl. Only swirl above a small threshold is reinforced, so
    // faint leftover eddies die out instead of being pumped up forever.
    // Worked out at cell centres, then shared out to the faces either side.
    const eps = this.vorticity * dt;
    const floor = 0.5;
    if (eps <= 0) return;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        // Centre velocities of the cells left / right and above / below.
        const vr = v[k + 1] + v[k + 1 - w], vl = v[k - 1] + v[k - 1 - w];
        const ud = u[k + w] + u[k + w - 1], uu = u[k - w] + u[k - w - 1];
        curl[k] = 0.25 * ((vr - vl) - (ud - uu));
      }
    }
    fx.fill(0);
    fy.fill(0);
    for (let j = 2; j < ny; j++) {
      for (let i = 2, k = 2 + w * j; i < nx; i++, k++) {
        const cr = curl[k + 1], cl = curl[k - 1], cd = curl[k + w], cu = curl[k - w];
        const gx = (cr < 0 ? -cr : cr) - (cl < 0 ? -cl : cl);
        const gy = (cd < 0 ? -cd : cd) - (cu < 0 ? -cu : cu);
        const len2 = gx * gx + gy * gy;
        const ck = curl[k];
        const strong = ck > floor ? ck - floor : ck < -floor ? ck + floor : 0;
        if (strong === 0 || len2 < 1e-8) continue;
        const c = strong * eps / Math.sqrt(len2);
        fx[k] = gy * c;
        fy[k] = -gx * c;
      }
    }
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        if (i < nx) u[k] += 0.5 * (fx[k] + fx[k + 1]);
        if (j < ny) v[k] += 0.5 * (fy[k] + fy[k + w]);
      }
    }
  }

  // Implicit viscous diffusion, solved with Gauss-Seidel, on the inner
  // faces; the edges stay as the boundary sets them.
  diffuseVelocity(dt) {
    const a = dt * this.visc;
    if (a <= 0) return;
    const { nx, ny, w, u, v, u0, v0 } = this;
    u0.set(u);
    v0.set(v);
    const c = 1 / (1 + 4 * a);
    for (let it = 0; it < this.viscIters; it++) {
      for (let j = 1; j <= ny; j++) {
        for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
          if (i < nx) u[k] = (u0[k] + a * (u[k - 1] + u[k + 1] + u[k - w] + u[k + w])) * c;
          if (j < ny) v[k] = (v0[k] + a * (v[k - 1] + v[k + 1] + v[k - w] + v[k + w])) * c;
        }
      }
      this.tangentialBoundary();
    }
  }

  // Sets the divergence of every cell to stiffness * (rho - 1) + zones:
  // squeezed fluid expands, stretched fluid contracts, and zones push the
  // balance one way or the other. The pressure that does it is solved to a
  // tolerance every step, warm-started from the last one.
  project() {
    const { nx, ny, w, u, v, p, div, rho, source } = this;
    const kappa = this.stiffness;

    // A sealed box can't gain or lose fluid, so then the target divergence
    // is balanced to zero overall.
    let offset = 0;
    if (this.walls && !this.hasWindows) {
      let sum = 0;
      for (let j = 1; j <= ny; j++) {
        for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) sum += kappa * (rho[k] - 1) + source[k];
      }
      offset = sum / (nx * ny);
    }
    // Right-hand side: target divergence minus the current one.
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        div[k] = kappa * (rho[k] - 1) + source[k] - offset - (u[k] - u[k - 1] + v[k] - v[k - w]);
      }
    }

    this.pressure.setBoundary(this.kindTop, this.kindBottom, this.kindLeft, this.kindRight);
    this.pressure.solve(p, div);

    // Subtract the pressure gradient on every face between two cells, and
    // on the outlet edges, where the pressure outside is ambient (0).
    const { kindTop, kindBottom, kindLeft, kindRight } = this;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        if (i < nx) u[k] -= p[k + 1] - p[k];
        if (j < ny) v[k] -= p[k + w] - p[k];
      }
    }
    for (let i = 1; i <= nx; i++) {
      if (kindTop[i] === OUTLET) v[i] -= p[i + w];
      if (kindBottom[i] === OUTLET) v[i + w * ny] += p[i + w * ny];
    }
    for (let j = 1; j <= ny; j++) {
      if (kindLeft[j] === OUTLET) u[w * j] -= p[1 + w * j];
      if (kindRight[j] === OUTLET) u[nx + w * j] += p[nx + w * j];
    }

    // Cap the speed so nothing can run away.
    const max = this.maxSpeed;
    for (let k = 0; k < u.length; k++) {
      if (u[k] > max) u[k] = max; else if (u[k] < -max) u[k] = -max;
      if (v[k] > max) v[k] = max; else if (v[k] < -max) v[k] = -max;
    }
    this.tangentialBoundary();
  }

  // Semi-Lagrangian advection of the velocity: each inner face looks back
  // along the flow and takes the velocity it finds there.
  advectVelocity(dt) {
    const { nx, ny, w, u, v, u0, v0 } = this;
    u0.set(u);
    v0.set(v);
    const xmax = nx + 0.5, ymax = ny + 0.5;
    const clampX = (x) => (x < 0.5 ? 0.5 : x > xmax ? xmax : x);
    const clampY = (y) => (y < 0.5 ? 0.5 : y > ymax ? ymax : y);
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        if (i < nx) {
          const vv = 0.25 * (v0[k] + v0[k + 1] + v0[k - w] + v0[k + 1 - w]);
          u[k] = this.sampleU(u0, clampX(i + 0.5 - dt * u0[k]), clampY(j - dt * vv));
        }
        if (j < ny) {
          const uu = 0.25 * (u0[k] + u0[k - 1] + u0[k + w] + u0[k - 1 + w]);
          v[k] = this.sampleV(v0, clampX(i - dt * uu), clampY(j + 0.5 - dt * v0[k]));
        }
      }
    }
  }

  // forces -> viscosity -> self-advection -> projection. A single projection
  // at the end keeps the field the particles see divergence-free.
  step(dt) {
    this.updateBoundaryKinds(dt);
    this.addForces(dt);
    this.applyWindShear(dt);
    this.velocityBoundary();
    this.diffuseVelocity(dt);
    this.advectVelocity(dt);
    this.velocityBoundary();
    this.project();
    this.updateDensity(dt);
  }

  // Continuity: carry rho with the flow, then compress or expand it by the
  // divergence of the velocity.
  updateDensity(dt) {
    const { nx, ny, w, u, v, rho, rho0 } = this;
    rho0.set(rho);
    const xmax = nx + 0.5, ymax = ny + 0.5;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        let x = i - dt * 0.5 * (u[k] + u[k - 1]), y = j - dt * 0.5 * (v[k] + v[k - w]);
        if (x < 0.5) x = 0.5; else if (x > xmax) x = xmax;
        if (y < 0.5) y = 0.5; else if (y > ymax) y = ymax;
        const d = u[k] - u[k - 1] + v[k] - v[k - w];
        const r = this.sampleC(rho0, x, y) * Math.exp(-d * dt);
        rho[k] = r < 0.2 ? 0.2 : r > 5 ? 5 : r;
      }
    }
    this.scalarBoundary(rho, 1);
  }
}

/*
 * The pressure Poisson equation, A p = b, solved by conjugate gradients
 * with a multigrid V-cycle as the preconditioner (McAdams et al., 2010).
 *
 * A is the five-point Laplacian with the sign flipped so it is positive:
 * (A p)_k = diag_k p_k - sum over neighbours of p. Faces between two cells
 * couple them; an outlet edge is a neighbour fixed at 0 (so it only adds to
 * the diagonal); a wall or inlet edge has no term at all. Each coarser level
 * is the Galerkin product P^T A P, with P copying a coarse cell into the 2x2
 * fine cells it covers, so the preconditioner is symmetric and CG stays
 * valid however the edges are mixed. A sealed box has no outlets, A is then
 * singular (any constant can be added to p), and the solve works in the
 * space of zero-mean pressures.
 */
class PressureSolver {
  constructor(nx, ny) {
    this.levels = [];
    for (;;) {
      const w = nx + 2, n = w * (ny + 2);
      const f = () => new Float32Array(n);
      this.levels.push({ nx, ny, w, diag: f(), cE: f(), cS: f(), x: f(), b: f(), r: f() });
      if (nx <= 4 || ny <= 4 || this.levels.length >= 12) break;
      nx = (nx + 1) >> 1;
      ny = (ny + 1) >> 1;
    }
    const n = this.levels[0].x.length;
    this.r = new Float32Array(n); // CG residual, direction, A * direction
    this.s = new Float32Array(n);
    this.q = new Float32Array(n);
    this.singular = false;
    this.key = '';
    this.tolerance = 5e-3; // largest divergence error left, cells per second
    this.maxIters = 16;   // a frame where the edges change can need more; the next one finishes
    this.sweeps = 1;      // smoothing passes each way per level
    this.iters = 0;       // what the last solve took
    this.residual = 0;
  }

  // Rebuilds the operator on every level when the edges change.
  setBoundary(kindTop, kindBottom, kindLeft, kindRight) {
    const key = String.fromCharCode(...kindTop, ...kindBottom, ...kindLeft, ...kindRight);
    if (key === this.key) return;
    this.key = key;
    const L = this.levels[0], { nx, ny, w, diag, cE, cS } = L;
    diag.fill(0); cE.fill(0); cS.fill(0);
    let dirichlet = false;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        let d = 4;
        if (i === 1 && kindLeft[j] !== OUTLET) d--;
        if (i === nx && kindRight[j] !== OUTLET) d--;
        if (j === 1 && kindTop[i] !== OUTLET) d--;
        if (j === ny && kindBottom[i] !== OUTLET) d--;
        diag[k] = d;
        if (i < nx) cE[k] = 1;
        if (j < ny) cS[k] = 1;
      }
    }
    for (const kind of [kindTop, kindBottom, kindLeft, kindRight]) {
      if (kind.some((k, i) => i > 0 && i < kind.length - 1 && k === OUTLET)) dirichlet = true;
    }
    this.singular = !dirichlet;

    // Galerkin coarsening. A coupling between two fine cells in the same
    // coarse cell cancels out of the coarse diagonal (twice, once for each
    // side); one between different coarse cells adds to their coupling.
    for (let l = 1; l < this.levels.length; l++) {
      const F = this.levels[l - 1], C = this.levels[l];
      C.diag.fill(0); C.cE.fill(0); C.cS.fill(0);
      for (let j = 1; j <= F.ny; j++) {
        const J = (j + 1) >> 1;
        for (let i = 1, k = 1 + F.w * j; i <= F.nx; i++, k++) {
          const K = ((i + 1) >> 1) + C.w * J;
          C.diag[K] += F.diag[k];
          if (F.cE[k]) { if (i & 1) C.diag[K] -= 2 * F.cE[k]; else C.cE[K] += F.cE[k]; }
          if (F.cS[k]) { if (j & 1) C.diag[K] -= 2 * F.cS[k]; else C.cS[K] += F.cS[k]; }
        }
      }
    }
  }

  // out = A x on level L.
  apply(L, x, out) {
    const { nx, ny, w, diag, cE, cS } = L;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        out[k] = diag[k] * x[k] - cE[k] * x[k + 1] - cE[k - 1] * x[k - 1] - cS[k] * x[k + w] - cS[k - w] * x[k - w];
      }
    }
  }

  // One Gauss-Seidel pass over the cells of one colour of a checkerboard.
  relax(L, colour) {
    const { nx, ny, w, diag, cE, cS, x, b } = L;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1 + ((j + colour + 1) & 1), k = i + w * j; i <= nx; i += 2, k += 2) {
        const d = diag[k];
        if (d > 0) x[k] = (b[k] + cE[k] * x[k + 1] + cE[k - 1] * x[k - 1] + cS[k] * x[k + w] + cS[k - w] * x[k - w]) / d;
      }
    }
  }

  // Approximately solves level l from its b, starting at x = 0. Smoothing
  // is red-black going down and black-red coming up, which keeps the
  // whole cycle symmetric.
  vcycle(l) {
    const L = this.levels[l];
    L.x.fill(0);
    if (l === this.levels.length - 1) {
      for (let s = 0; s < 20; s++) { this.relax(L, 0); this.relax(L, 1); this.relax(L, 1); this.relax(L, 0); }
      return;
    }
    for (let s = 0; s < this.sweeps; s++) { this.relax(L, 0); this.relax(L, 1); }
    this.apply(L, L.x, L.r);
    const C = this.levels[l + 1];
    C.b.fill(0);
    for (let j = 1; j <= L.ny; j++) {
      const J = (j + 1) >> 1;
      for (let i = 1, k = 1 + L.w * j; i <= L.nx; i++, k++) {
        C.b[((i + 1) >> 1) + C.w * J] += L.b[k] - L.r[k];
      }
    }
    this.vcycle(l + 1);
    for (let j = 1; j <= L.ny; j++) {
      const J = (j + 1) >> 1;
      for (let i = 1, k = 1 + L.w * j; i <= L.nx; i++, k++) L.x[k] += C.x[((i + 1) >> 1) + C.w * J];
    }
    for (let s = 0; s < this.sweeps; s++) { this.relax(L, 1); this.relax(L, 0); }
  }

  // z = M^-1 r: one V-cycle on the residual.
  precondition(r) {
    const L = this.levels[0];
    L.b.set(r);
    this.vcycle(0);
    if (this.singular) this.removeMean(L.x);
    return L.x;
  }

  removeMean(x) {
    const { nx, ny, w } = this.levels[0];
    let sum = 0;
    for (let j = 1; j <= ny; j++) for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) sum += x[k];
    const m = sum / (nx * ny);
    for (let j = 1; j <= ny; j++) for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) x[k] -= m;
  }

  dot(a, b) {
    const { nx, ny, w } = this.levels[0];
    let sum = 0;
    for (let j = 1; j <= ny; j++) for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) sum += a[k] * b[k];
    return sum;
  }

  maxAbs(a) {
    const { nx, ny, w } = this.levels[0];
    let m = 0;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) { const x = a[k] < 0 ? -a[k] : a[k]; if (x > m) m = x; }
    }
    return m;
  }

  // Solves A p = b in place, starting from the p passed in.
  solve(p, b) {
    const L = this.levels[0], { r, s, q } = this;
    if (this.singular) this.removeMean(p);
    this.apply(L, p, r);
    for (let k = 0; k < r.length; k++) r[k] = b[k] - r[k];
    if (this.singular) this.removeMean(r);
    let res = this.maxAbs(r), it = 0;
    if (res > this.tolerance) {
      s.set(this.precondition(r));
      let rz = this.dot(r, s);
      for (; it < this.maxIters && res > this.tolerance; it++) {
        this.apply(L, s, q);
        const sq = this.dot(s, q);
        if (!(sq > 0)) break;
        const alpha = rz / sq;
        for (let k = 0; k < p.length; k++) { p[k] += alpha * s[k]; r[k] -= alpha * q[k]; }
        res = this.maxAbs(r);
        if (res <= this.tolerance) { it++; break; }
        const z = this.precondition(r);
        const rzNew = this.dot(r, z);
        const beta = rzNew / rz;
        rz = rzNew;
        for (let k = 0; k < s.length; k++) s[k] = z[k] + beta * s[k];
      }
    }
    this.iters = it;
    this.residual = res;
  }
}
