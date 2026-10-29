'use strict';

/*
 * Stable-fluids solver (Stam, 1999) on a uniform grid with one ghost cell on
 * every side. Grid units throughout: cell (i, j) has its centre at (i, j),
 * velocities are in cells per second.
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
    this.pressureIters = 24;
    this.viscIters = 12;
    this.updateBoundaryKinds(0);
  }

  reset() {
    for (const f of [this.u, this.v, this.u0, this.v0, this.p, this.div, this.curl]) f.fill(0);
    this.rho.fill(1);
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

  // Inlets hold the wind velocity. Outlets pass outflow through freely but
  // don't copy inflow (copying backflow is a classic runaway); air can
  // still be drawn in there by the pressure. Outside air streaming past an
  // outlet drags its tangential component along with the wind. Walls: the
  // normal component is mirrored (no flow through the wall), the tangential
  // one slips freely.
  velocityBoundary() {
    const { nx, ny, w, u, v, kindTop, kindBottom, kindLeft, kindRight, windU: wu, windV: wv } = this;
    const windy = wu !== 0 || wv !== 0;
    const last = w * (ny + 1);
    for (let i = 1; i <= nx; i++) {
      const t = i, b = last + i;
      const kt = kindTop[i], kb = kindBottom[i];
      if (kt === WALL) { u[t] = u[t + w]; v[t] = -v[t + w]; }
      else if (kt === INLET) { u[t] = wu; v[t] = wv; }
      else { u[t] = windy ? wu : u[t + w]; v[t] = Math.min(v[t + w], 0); }
      if (kb === WALL) { u[b] = u[b - w]; v[b] = -v[b - w]; }
      else if (kb === INLET) { u[b] = wu; v[b] = wv; }
      else { u[b] = windy ? wu : u[b - w]; v[b] = Math.max(v[b - w], 0); }
    }
    for (let j = 1; j <= ny; j++) {
      const l = w * j, r = l + nx + 1;
      const kl = kindLeft[j], kr = kindRight[j];
      if (kl === WALL) { u[l] = -u[l + 1]; v[l] = v[l + 1]; }
      else if (kl === INLET) { u[l] = wu; v[l] = wv; }
      else { u[l] = Math.min(u[l + 1], 0); v[l] = windy ? wv : v[l + 1]; }
      if (kr === WALL) { u[r] = -u[r - 1]; v[r] = v[r - 1]; }
      else if (kr === INLET) { u[r] = wu; v[r] = wv; }
      else { u[r] = Math.max(u[r - 1], 0); v[r] = windy ? wv : v[r - 1]; }
    }
    this.corners(u);
    this.corners(v);
  }

  // Turbulent shear at openings: the outside wind streaming past an opening
  // drags the air just inside it along. The first two cells inside each
  // outlet relax their along-the-wall velocity towards the wind's.
  applyWindShear(dt) {
    const { nx, ny, w, u, v, kindTop, kindBottom, kindLeft, kindRight, windU: wu, windV: wv } = this;
    if (wu === 0 && wv === 0) return;
    const f1 = 1 - Math.exp(-this.shear * dt), f2 = 0.5 * f1;
    for (let i = 1; i <= nx; i++) {
      if (kindTop[i] === OUTLET) {
        const k = i + w;
        u[k] += (wu - u[k]) * f1;
        u[k + w] += (wu - u[k + w]) * f2;
      }
      if (kindBottom[i] === OUTLET) {
        const k = i + w * ny;
        u[k] += (wu - u[k]) * f1;
        u[k - w] += (wu - u[k - w]) * f2;
      }
    }
    for (let j = 1; j <= ny; j++) {
      if (kindLeft[j] === OUTLET) {
        const k = 1 + w * j;
        v[k] += (wv - v[k]) * f1;
        v[k + 1] += (wv - v[k + 1]) * f2;
      }
      if (kindRight[j] === OUTLET) {
        const k = nx + w * j;
        v[k] += (wv - v[k]) * f1;
        v[k - 1] += (wv - v[k - 1]) * f2;
      }
    }
  }

  // Pressure: ambient (0) at outlets, zero-gradient at walls and at wind
  // inlets (where the velocity is prescribed instead).
  pressureBoundary(p) {
    const { nx, ny, w, kindTop, kindBottom, kindLeft, kindRight } = this;
    const last = w * (ny + 1);
    for (let i = 1; i <= nx; i++) {
      p[i] = kindTop[i] === OUTLET ? 0 : p[i + w];
      p[last + i] = kindBottom[i] === OUTLET ? 0 : p[last + i - w];
    }
    for (let j = 1; j <= ny; j++) {
      const r = w * j;
      p[r] = kindLeft[j] === OUTLET ? 0 : p[r + 1];
      p[r + nx + 1] = kindRight[j] === OUTLET ? 0 : p[r + nx];
    }
    this.corners(p);
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

  // Nudges the velocity under a soft circular brush towards (vx, vy).
  splatVelocity(cx, cy, r, vx, vy, strength) {
    const { nx, ny, w, u, v } = this;
    const r2 = r * r;
    const i0 = Math.max(1, Math.floor(cx - r)), i1 = Math.min(nx, Math.ceil(cx + r));
    const j0 = Math.max(1, Math.floor(cy - r)), j1 = Math.min(ny, Math.ceil(cy + r));
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const dx = i - cx, dy = j - cy, d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;
        const k = i + w * j;
        const f = strength * Math.exp(-3 * d2 / r2);
        u[k] += (vx - u[k]) * f;
        v[k] += (vy - v[k]) * f;
      }
    }
  }

  // ------------------------------------------------------------------ solver

  addForces(dt) {
    const { nx, ny, w, u, v, curl } = this;

    // Vorticity confinement: push along N x omega, where N points towards
    // stronger swirl. Only swirl above a small threshold is reinforced, so
    // faint leftover eddies die out instead of being pumped up forever.
    const eps = this.vorticity * dt;
    const floor = 0.5;
    if (eps > 0) {
      for (let j = 1; j <= ny; j++) {
        for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
          curl[k] = 0.5 * ((v[k + 1] - v[k - 1]) - (u[k + w] - u[k - w]));
        }
      }
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
          u[k] += gy * c;
          v[k] -= gx * c;
        }
      }
    }
  }

  // Implicit viscous diffusion, solved with Gauss-Seidel.
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
          u[k] = (u0[k] + a * (u[k - 1] + u[k + 1] + u[k - w] + u[k + w])) * c;
          v[k] = (v0[k] + a * (v[k - 1] + v[k + 1] + v[k - w] + v[k + w])) * c;
        }
      }
      this.velocityBoundary();
    }
  }

  // Sets the divergence of the velocity to stiffness * (rho - 1) + zones:
  // squeezed fluid expands, stretched fluid contracts, and zones push the
  // balance one way or the other. SOR with the previous pressure as a warm
  // start, so a few iterations per frame are enough.
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
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        div[k] = -0.5 * (u[k + 1] - u[k - 1] + v[k + w] - v[k - w]) + kappa * (rho[k] - 1) + source[k] - offset;
      }
    }
    this.pressureBoundary(p);

    const omega = 1.75;
    for (let it = 0; it < this.pressureIters; it++) {
      for (let j = 1; j <= ny; j++) {
        for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
          p[k] += omega * ((div[k] + p[k - 1] + p[k + 1] + p[k - w] + p[k + w]) * 0.25 - p[k]);
        }
      }
      this.pressureBoundary(p);
    }

    // Subtract the pressure gradient, and cap the speed so nothing can run
    // away.
    const max2 = this.maxSpeed * this.maxSpeed;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        const uk = u[k] - 0.5 * (p[k + 1] - p[k - 1]);
        const vk = v[k] - 0.5 * (p[k + w] - p[k - w]);
        const s2 = uk * uk + vk * vk;
        if (s2 > max2) {
          const f = this.maxSpeed / Math.sqrt(s2);
          u[k] = uk * f;
          v[k] = vk * f;
        } else {
          u[k] = uk;
          v[k] = vk;
        }
      }
    }
    this.velocityBoundary();
  }

  // Semi-Lagrangian advection of d0 into d along (uu, vv).
  advect(d, d0, uu, vv, dt) {
    const { nx, ny, w } = this;
    const xmax = nx + 0.5, ymax = ny + 0.5;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        let x = i - dt * uu[k], y = j - dt * vv[k];
        if (x < 0.5) x = 0.5; else if (x > xmax) x = xmax;
        if (y < 0.5) y = 0.5; else if (y > ymax) y = ymax;
        const i0 = x | 0, j0 = y | 0;
        const s1 = x - i0, t1 = y - j0, s0 = 1 - s1, t0 = 1 - t1;
        const k0 = i0 + w * j0;
        d[k] = s0 * (t0 * d0[k0] + t1 * d0[k0 + w]) + s1 * (t0 * d0[k0 + 1] + t1 * d0[k0 + 1 + w]);
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

    this.u0.set(this.u);
    this.v0.set(this.v);
    this.advect(this.u, this.u0, this.u0, this.v0, dt);
    this.advect(this.v, this.v0, this.u0, this.v0, dt);
    this.velocityBoundary();
    this.project();

    this.updateDensity(dt);
  }

  // Continuity: carry rho with the flow, then compress or expand it by the
  // divergence of the velocity.
  updateDensity(dt) {
    const { nx, ny, w, u, v, rho } = this;
    this.rho0.set(rho);
    this.advect(rho, this.rho0, u, v, dt);
    for (let j = 1; j <= ny; j++) {
      for (let i = 1, k = 1 + w * j; i <= nx; i++, k++) {
        const d = 0.5 * (u[k + 1] - u[k - 1] + v[k + w] - v[k - w]);
        const r = rho[k] * Math.exp(-d * dt);
        rho[k] = r < 0.2 ? 0.2 : r > 5 ? 5 : r;
      }
    }
    this.scalarBoundary(rho, 1);
  }
}
