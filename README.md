# fluids

**[▶ Run it](https://boris-volkov.github.io/fluids/)**

A fluid seen from above that you can push around, blow fans through, squeeze and open windows in, made visible by up to 150,000 particles carried along with it.

The particles start out as an evenly spaced grid, and that is the point of them: they are markers, not molecules, and every way the grid gets stretched, folded and wound into spirals is something the flow did. Nothing is ever created or destroyed inside the fluid. Particles leave only by being carried off an open edge, and arrive only with fluid flowing in from outside.

The tools:

**Stir.** Drag to push the fluid.

**Low and high pressure.** A low-pressure zone draws fluid in and compresses it; a high-pressure zone pushes it out. Either way the flow stops once the density inside balances the zone, and the fluid relaxes back once the zone is gone.

**Fans.** Click to place one and drag to aim it. A fan keeps pushing the fluid inside it the way it points, so it blows a jet out of the front and draws fluid in from behind and from the sides, shedding a pair of eddies as it goes. Shift-drag a fan to re-aim it.

**Walls and windows.** By default the edges are open, at ambient pressure, so fluid can come and go. Close them into a box, then cut windows into the walls. Add a wind and a room with only windward openings breathes in and out through them.

Viscosity runs from inviscid to treacle, and the swirl slider keeps small eddies from being smeared away.

## How it works

`sim.js` is Jos Stam's stable-fluids solver (1999): forces, implicit viscous diffusion, semi-Lagrangian advection, then a pressure projection that keeps the velocity field divergence-free. On top of it go fans, vorticity confinement, and a weak compressibility, where a carried density field lets squeezed fluid expand again and stretched fluid contract.

The grid is staggered (a MAC grid): pressure sits at the centre of each cell and each velocity component on the faces between cells, so a cell's divergence is exactly what flows through its four faces. The pressure is solved to a tolerance every step, by conjugate gradients preconditioned with a multigrid V-cycle (McAdams et al., 2010), warm-started from the step before. The fluid that comes out the other side carries no more into a cell than out of it, to within 0.005 cells per second, and the effect of opening a window reaches the far wall in the same step instead of creeping across the room a few cells at a time.

`app.js` moves the particles through that velocity field with a midpoint (RK2) step and draws them as instanced quads in WebGL2, stretched into streaks along their velocity if you ask for it. Browsers without WebGL2 fall back to a 2D canvas.

No build step and no dependencies — open `index.html`.
