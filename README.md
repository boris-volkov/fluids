# fluids

**[▶ Run it](https://boris-volkov.github.io/fluids/)**

A fluid you can push around, heat, squeeze and open windows in, made visible by up to 150,000 particles carried along with it.

The particles start out as an evenly spaced grid, and that is the point of them: they are markers, not molecules, and every way the grid gets stretched, folded and wound into spirals is something the flow did. Nothing is ever created or destroyed inside the fluid. Particles leave only by being carried off an open edge, and arrive only with fluid flowing in from outside.

The tools:

**Stir.** Drag to push the fluid.

**Low and high pressure.** A low-pressure zone draws fluid in and compresses it; a high-pressure zone pushes it out. Either way the flow stops once the density inside balances the zone, and the fluid relaxes back once the zone is gone.

**Heat and cold.** Warm fluid rises, cold fluid sinks, and the temperature is carried along with the flow and fades as it cools.

**Walls and windows.** By default the edges are open, at ambient pressure, so fluid can come and go. Close them into a box, then cut windows into the walls. Add a wind and a room with only windward openings breathes in and out through them.

Viscosity runs from inviscid to treacle, and the swirl slider keeps small eddies from being smeared away.

## How it works

`sim.js` is Jos Stam's stable-fluids solver (1999) on a uniform grid: forces, implicit viscous diffusion, semi-Lagrangian advection, then a pressure projection that keeps the velocity field divergence-free. On top of it go a buoyant temperature field, vorticity confinement, and a weak compressibility, where a carried density field lets squeezed fluid expand again and stretched fluid contract.

`app.js` moves the particles through that velocity field with a midpoint (RK2) step and draws them as instanced quads in WebGL2, stretched into streaks along their velocity if you ask for it. Browsers without WebGL2 fall back to a 2D canvas.

No build step and no dependencies — open `index.html`.
