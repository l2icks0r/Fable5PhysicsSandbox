# Real-Time Rigid-Body Physics Sandbox

A 3D rigid-body physics sandbox built **from scratch with zero libraries** — no
Three.js, no Cannon, no Rapier, no math library, no build framework. Every line
of the projection math, the rasterizer, the collision detection, and the
constraint solver is here to be read.

Drop **cubes, spheres, and tetrahedra** onto a floor and watch them tumble,
stack, roll, and settle. Then flip two switches live, mid-simulation:

- the **renderer** — a hand-written CPU software rasterizer, WebGL, or WebGPU;
- the **physics engine** — the same simulation written twice, once in
  JavaScript and once in C++ compiled to WebAssembly.

It runs by opening `index.html` directly from disk — no server, no toolchain,
nothing to install. The whole thing is four small JavaScript files, three
generated/support files, and one HTML shell.

The goal is pedagogical: it is written to be **read and learned from**. If you
want to understand how a rigid-body engine actually works — SAT collision,
contact manifolds, sequential-impulse solving, sleeping islands — the code is
commented for exactly that, and this document walks through the ideas.

---

## Table of contents

- [Quick start](#quick-start)
- [Controls](#controls)
- [Features](#features)
- [Project layout](#project-layout)
- [The physics engine](#the-physics-engine)
  - [Rigid-body state and the isotropic-inertia shortcut](#rigid-body-state-and-the-isotropic-inertia-shortcut)
  - [Integration and the fixed timestep](#integration-and-the-fixed-timestep)
  - [Broad phase: the spatial hash](#broad-phase-the-spatial-hash)
  - [Narrow phase: collision detection](#narrow-phase-collision-detection)
  - [Contact solving: sequential impulses](#contact-solving-sequential-impulses)
  - [The floor, and the edge cases at its edge](#the-floor-and-the-edge-cases-at-its-edge)
  - [Sleeping and waking](#sleeping-and-waking)
  - [Rolling](#rolling)
- [The renderers](#the-renderers)
  - [The renderer interface](#the-renderer-interface)
  - [Software rasterizer](#software-rasterizer)
  - [WebGL](#webgl)
  - [WebGPU](#webgpu)
- [Two physics engines: JavaScript and WebAssembly](#two-physics-engines-javascript-and-webassembly)
- [The camera](#the-camera)
- [Building the WebAssembly module](#building-the-webassembly-module)
- [Performance notes](#performance-notes)
- [Credits and license](#credits-and-license)

---

## Quick start

**Just run it:** open `index.html` in a modern browser (Chrome or Edge for the
full experience, including WebGPU). That's it. Left-click on the floor to drop a
body.

You do **not** need any tools to run the project. You only need the
[Emscripten](https://emscripten.org) SDK if you want to *rebuild* the C++
physics engine after editing it — the compiled result is already checked in.

---

## Controls

| Input | Action |
|---|---|
| **Left click** / hold **Space** | Drop a body at the pointer (Space rains them) |
| **Shift + mouse** | Look around (first-person) |
| **Ctrl + mouse** | Orbit the scene around its vertical axis |
| **W A S D** | Fly forward / strafe |
| **Arrow ↑ ↓** | Pedestal (move straight up/down) |
| **Arrow ← →** | Strafe left/right |
| Mouse wheel / **+ −** | Zoom (dolly) |
| **B** | Bounce the floor — jolt every body upward |
| **C** | Clear the scene |
| **T** | Cycle shape: cube → sphere → tetrahedron |
| **R** | Cycle renderer: software → WebGL → WebGPU |
| **P** | Cycle physics engine: JavaScript → WebAssembly |

On-screen buttons mirror bounce, clear, shape, physics, renderer, and
fullscreen for touch devices.

---

## Features

- **Three interchangeable renderers**, switchable live, all producing a
  pixel-nearly-identical image of the same scene:
  - a **software rasterizer** — a full 3D pipeline (projection, clipping,
    z-buffered scanline fill, flat shading) implemented by hand on a 2D canvas,
    including *analytic ray-traced spheres*;
  - **WebGL 2** — the classic GPU pipeline, one instanced draw call per shape;
  - **WebGPU** — the modern explicit GPU API, the same scene through pipelines
    and command encoders.
- **Two interchangeable physics engines**, switchable live mid-scene:
  - the reference engine in **JavaScript**;
  - a line-for-line port in **C++ compiled to WebAssembly**, ~3.5× faster.
- **Three body shapes** — cube, sphere, and regular tetrahedron — colliding with
  each other and the floor in any combination.
- **Beach-ball spheres** with a procedural wedge pattern so rotation is visible,
  smoothly shaded.
- A **fly camera** with look, orbit, pedestal, and zoom.
- **Sleeping** so large settled piles cost almost nothing, and a live **FPS
  counter** to watch the engines and renderers compete.

---

## Project layout

```
index.html              The shell: markup, styles, and the script tags.
simulation.js           Physics (JS engine), camera, input, main loop, and the
                        renderer/engine registries. The heart of the project.
renderer-software.js    CPU software rasterizer + its render cache.
renderer-webgl.js       WebGL 2 renderer.
renderer-webgpu.js      WebGPU renderer.
physics.cpp             The C++ physics engine (source for the WASM build).
physics-wasm.js         JS wrapper that loads the WASM module and syncs state.
physics-wasm-bin.js     GENERATED: physics.wasm embedded as base64.
physics.wasm            GENERATED: the compiled module (intermediate artifact).
build-wasm.bat          Rebuilds the two generated files from physics.cpp.
```

Everything is loaded with plain `<script>` tags (no modules, no bundler),
which is what lets the page run straight from the filesystem. Files share one
global scope: `simulation.js` owns the vector math, constants, and the `bodies`
array, and the renderer/engine files read those globals. The load order in
`index.html` matters — renderers and the WASM support load before
`simulation.js`, which wires everything together and starts the loop.

---

## The physics engine

This is a real-time **impulse-based rigid-body simulator**. If you have seen
Box2D or Bullet, the architecture will be familiar; it is the same pipeline,
kept single-threaded and single-manifold for clarity:

```
for each fixed timestep:
    integrate forces  ->  broad phase  ->  narrow phase  ->  solve contacts  ->  integrate velocities
```

The JavaScript implementation lives in `simulation.js` (`physStep` and the
functions above it); the C++ port in `physics.cpp` mirrors it exactly. What
follows describes both.

### Rigid-body state and the isotropic-inertia shortcut

Each body stores:

- **position** and **linear velocity** (`vec3`),
- **orientation** as a unit **quaternion**, and **angular velocity** (`vec3`),
- inverse mass and inverse inertia.

Orientation is a quaternion rather than a matrix or Euler angles because
quaternions integrate cleanly and never hit gimbal lock. Each step we cache the
body's three local axes (the columns of its rotation matrix) so collision code
can transform points without re-deriving the matrix.

There is one lovely simplification the whole solver leans on. A rigid body's
resistance to rotation is in general a **3×3 inertia tensor**, and using it
means rotating it into world space every time you touch it. But all three of
our shapes — cube, sphere, and **regular tetrahedron** — have an **isotropic**
inertia tensor: a scalar times the identity matrix. (For the tetrahedron this
is not a coincidence — every Platonic solid's symmetry forces its inertia
ellipsoid to be a sphere.) An isotropic tensor is unchanged by rotation, so the
world-space inverse inertia stays a **single number** forever. That is why you
see `invI` as a scalar throughout the code instead of a matrix. A square
pyramid would *not* have this property and would have forced a full tensor
through the entire solver — which is exactly why the third shape is a
tetrahedron.

### Integration and the fixed timestep

Velocities and positions advance with **semi-implicit (symplectic) Euler**:
gravity updates velocity first, then velocity updates position. That ordering is
noticeably more stable for constraint systems than plain Euler. Orientation
integrates via the quaternion derivative `q' = ½ · ω · q` (angular velocity as a
pure quaternion), followed by renormalization to kill numerical drift.

The simulation runs on a **fixed 120 Hz timestep**, decoupled from the display
refresh rate by an accumulator: each animation frame adds the real elapsed time
to a bucket and spends it in fixed 1/120 s slices. Fixed steps are essential —
an impulse solver tuned at one timestep becomes bouncy or mushy at another. The
loop caps the number of sub-steps per frame so a slow frame slows the
simulation down gracefully instead of spiraling into a "spiral of death."

### Broad phase: the spatial hash

Testing every pair of bodies is O(n²). The **broad phase** cheaply rejects pairs
that cannot possibly touch. We use a **uniform spatial hash**: space is divided
into cubic cells the size of a body's bounding-sphere diameter, and each body is
filed into the cell containing its center. Two bodies can only overlap if they
share a cell or sit in adjacent ones, so each body only ever checks the 27 cells
around it. Only *awake* bodies initiate checks, so a sleeping pile is never even
scanned. In JavaScript this is a `Map`; in C++ it is a fixed open-addressed hash
table with per-body chain links — no allocation at all.

### Narrow phase: collision detection

For each candidate pair that survives the broad phase (bounding spheres actually
overlapping), the **narrow phase** determines whether — and *where* — they
touch, producing a set of **contact points**, each with a world position, a
contact normal, and a penetration depth. The solver downstream never looks at
shapes; it only consumes contacts, which is what lets one solver handle every
shape combination.

Dispatch is by shape pair:

**Sphere vs. sphere** is the simplest collision test in existence: compare the
distance between centers to the sum of radii. The contact normal is the line
between centers.

**Sphere vs. convex** (cube or tetrahedron) finds the **closest point on the
hull** to the sphere's center. Checking the faces the center lies in front of —
projecting onto the face when the projection lands inside the polygon, otherwise
falling back to the nearest point on that face's edges — covers face, edge, and
corner contacts with one piece of code. If the center is swallowed inside the
hull, it is pushed out through the nearest face.

**Convex vs. convex** (cube–cube, cube–tet, tet–tet) uses the **Separating Axis
Theorem (SAT)**. Two convex shapes are disjoint if and only if some axis exists
on which their projected shadows don't overlap. For polyhedra the candidate axes
are the **face normals of both shapes** plus the **cross products of their edge
directions**. If any axis separates them, there is no collision. Otherwise the
axis of *least* penetration defines the contact normal, and we build the contact
manifold two ways depending on which kind of axis won:

- **Face case** (a face normal penetrates least): the owning shape supplies the
  *reference face*; the other supplies its most anti-parallel *incident face*.
  The incident face is clipped against the side planes of the reference face
  (Sutherland–Hodgman polygon clipping), and every clipped vertex still below
  the reference face becomes a contact point. This yields the multi-point
  manifolds (up to four points) that let a flat face rest stably instead of
  wobbling on one corner.
- **Edge case** (an edge-cross axis penetrates least, and clearly so — face
  axes are favored to avoid flicker): the two nearest points between the
  supporting edges give a single contact point. This is the "two tumbling boxes
  clip corners" case that a vertex-only test would miss.

The same routine handles all three polyhedron pairings because a shape is just a
list of vertices, faces, edges, and unique edge directions (`makeConvexShape` /
`buildConvex` derives these, auto-correcting face winding so normals point
outward).

### Contact solving: sequential impulses

Given all the contacts for a step, the solver makes the bodies stop
interpenetrating and respond realistically. It uses **sequential impulses** (the
method popularized by Erin Catto's Box2D), iterated 10 times per step:

- **Normal constraint** — for each contact, compute the relative velocity at the
  contact point, including the rotational term `ω × r`, and apply an impulse
  along the normal to push it toward a target separating velocity. The impulse
  magnitude divides by the **effective mass** `1/(1/mₐ + 1/m_b + Iₐ⁻¹|rₐ×n|² +
  I_b⁻¹|r_b×n|²)`, which correctly accounts for how much of the push turns into
  spin rather than translation. Accumulated impulses are **clamped to be
  non-negative** — the classic trick that lets a later iteration walk back an
  earlier over-correction without ever *pulling* bodies together.
- **Restitution** (bounciness) — the pre-solve approach speed is remembered, and
  the target rebound is `−e · v_approach`, but only above a small threshold.
  Below it, bounce is killed, which is what lets bodies actually come to rest
  instead of jittering forever.
- **Friction** — two tangent directions each get an impulse that drives
  tangential velocity to zero, clamped inside the **Coulomb cone** `|j_t| ≤ μ·jₙ`.
  This produces sliding friction *and* the torque coupling that makes a spinning
  body veer or a rolling ball spin up.
- **Penetration recovery** — a **Baumgarte** bias folds a little
  position-correction velocity (proportional to penetration depth past a small
  slop) into the normal target, easing overlapping bodies apart over a few
  frames rather than teleporting them. The bias is capped so a deep overlap
  (e.g. a body spawned inside another) can't launch anything.

The materials are tuned like plastic dice on a wooden floor: low restitution
(0.22 on the floor, 0.12 between bodies) and moderate friction (0.45 / 0.35).

### The floor, and the edge cases at its edge

The floor is a finite plane, and getting bodies to fall off its edge *correctly*
took more care than the whole rest of the floor code. Contacts are generated
per **vertex** for polyhedra — but only for vertices actually above the floor
rectangle, so an overhanging corner correctly generates no support and the body
tips off. Two guards prevent a notorious class of bug where a body rolling near
the rim gets flung into the air:

1. a body with any vertex driven *deep* below the plane while over it has fallen
   past the edge and is now *under* the floor — a thin plane never pushes from
   below, so it gets no floor contact at all;
2. a submerged vertex closer to the plane's side boundary than to its surface is
   treated as past the edge, not under the floor.

A third contact type, **rim contacts**, supports a body *straddling* the edge:
wherever one of the body's edges crosses the floor boundary at or below the
surface, a contact is placed on the floor's actual rim line, so the body pivots
about the real edge (like a book tipping off a table) instead of sinking through
it. Spheres skip all of this — their floor test is a single nearest-point query
against the floor rectangle that handles the flat top, the rim, and the corners
in one shot.

### Sleeping and waking

A body that stays slow while touching something for half a second is put to
**sleep**: it is skipped entirely by integration and collision until something
disturbs it. This is what makes a 900-body pile essentially free to simulate.
Waking is careful — a fast body crashing in wakes a sleeper *and* whatever it
was resting on (one level of chain), and an unusually large solved impulse (well
above the quiet weight of a tall stack) counts as a shove and wakes bodies too.
Extra damping on near-still bodies helps pile jitter die out so things actually
reach the sleep threshold.

### Rolling

Rolling was not written — it **emerges** from the friction solver, which is the
satisfying part. The contact's relative velocity already includes `ω × r`; when
a sphere slides, friction applies a tangential impulse a radius below the
center, which torques it, spinning it up until the contact point's velocity
reaches zero. That *is* rolling without slipping. One addition was needed: a
perfect rolling sphere has zero contact velocity, so friction goes silent and it
would roll forever. A small **rolling-resistance** term bleeds angular velocity
while a sphere is touching something, so balls slow down and sleep like real
ones (which stop from contact-patch deformation that a point-contact model
lacks).

---

## The renderers

All three renderers draw the exact same scene and are switched at runtime. This
is the other half of the project's pedagogy: watching one scene rendered three
ways — a CPU rasterizer, a classic GPU pipeline, and a modern explicit GPU API —
makes the differences between them concrete.

### The renderer interface

`simulation.js` talks to renderers through a tiny interface, so none of the
physics, camera, or input code knows or cares which is active:

```
name              label shown on the renderer button
init(canvas)      set up; may be async; false/throw means unsupported
render(camMoved)  draw the current world
resize()          the canvas changed size
dispose()         release the GPU context (being switched out)
bodyWoke(b), bodyRemoved(b), sceneCleared()
                  cache-invalidation hooks — see below
```

A canvas is permanently bound to its first context type (2D, WebGL, or WebGPU),
so switching renderers swaps in a fresh `<canvas>` element. Unsupported
renderers (no WebGL2, no WebGPU) are skipped automatically when cycling.

Those last three hooks are the interesting bit. The GPU renderers implement them
as **no-ops** — they redraw the whole world every frame regardless. Only the
software renderer, which caches aggressively, does anything with them. This is a
deliberate teaching contrast: the caching machinery that dominates the software
renderer simply *evaporates* on the GPU.

### Software rasterizer

`renderer-software.js` is a complete 3D graphics pipeline written by hand,
drawing into an `ImageData` pixel buffer with a per-pixel **z-buffer**.

- **Polyhedra** are rasterized face by face. Each visible (front-facing) face is
  flat-shaded with a single directional light (`ambient + diffuse · max(0,
  N·L)`) and filled scanline by scanline. Depth uses **1/z**, which is affine in
  screen space across a flat face, so its gradient is computed once per face and
  walked incrementally — a classic rasterizer trick.
- **Spheres** are drawn a completely different and rather elegant way: **analytic
  ray-tracing**. For each pixel in the sphere's projected bounding box, the eye
  ray is intersected with the sphere; a hit gives an exact silhouette, a smooth
  **per-pixel** normal for lighting, and the correct depth. There is no sphere
  mesh at all. This is a nice illustration that a CPU can trace a shape directly
  where a GPU wants triangles. The **beach-ball wedge pattern** comes from the
  hit point expressed in the sphere's *own rotating frame*, so it tumbles with
  the body. (The bounding box is the exact projected **ellipse** — an off-axis
  sphere does not project to a circle — computed from the eye rays tangent to
  the sphere, which is what keeps edge-of-screen spheres from being clipped.)
- The **floor** is drawn with the antialiased 2D canvas API, and then also
  rasterized into the z-buffer as a depth-only pass, so a body that falls below
  the finite plane is correctly hidden by it.
- **Shadows** are the footprint of each body dropped straight down onto the
  floor (a convex hull for polyhedra, a circle for spheres), filled with
  height-faded translucent black.

Because filling pixels on the CPU is expensive, this renderer works hard to
avoid it. Everything that isn't moving — the floor, and every *sleeping* body —
is baked into cached color and depth layers; each frame only the moving bodies
are re-rasterized on top. When a body wakes or is removed, only the **dirty
screen rectangle** around it is rebuilt (this is what `bodyWoke` / `bodyRemoved`
track). While the camera is moving the cache is worthless, so the whole scene is
drawn directly until it stops. This caching is the single most complex part of
the project — and it is exactly the complexity the GPU renderers make vanish.

### WebGL

`renderer-webgl.js` uses **WebGL 2**. It builds one mesh per shape (a unit cube,
a UV sphere, a tetrahedron) and, each frame, uploads every body's position,
orientation quaternion, and color into a per-shape instance buffer, then issues
**one instanced draw call per shape** — three draws for the whole scene, however
many thousands of bodies. The vertex shader rotates each vertex by the instance
quaternion (`v + 2·cross(q.xyz, cross(q.xyz, v) + q.w·v)`) and does the flat
Lambert shade. Spheres get their own shader pair for smooth per-fragment
lighting and the procedural beach-ball pattern. The depth buffer resolves all
occlusion, so there is no sorting and no caching. Shadows reuse the meshes,
squashed flat onto the floor by the vertex shader, with the **stencil buffer**
ensuring each pixel is darkened only once where overlapping shadow triangles
pile up. Antialiasing is free (MSAA).

### WebGPU

`renderer-webgpu.js` draws the identical scene through the **modern explicit GPU
API**. The two GPU files are deliberately structured alike so they can be read
side by side; the difference is the *shape of the API*, not the result. WebGL is
a state machine you mutate between draws (enable blending, set the stencil op,
…). WebGPU front-loads all of that into immutable **pipeline** objects created
once at startup; each frame just records "set pipeline, set buffers, draw" into a
**command encoder** and submits it. Shaders are **WGSL** instead of GLSL, device
setup is **asynchronous**, and — a subtle gotcha spelled out in the projection
matrix — clip-space depth spans **[0, 1]** rather than WebGL's [−1, 1].
Rendering is 4× multisampled and resolved to the canvas each frame.

---

## Two physics engines: JavaScript and WebAssembly

The simulation exists **twice**: the reference implementation in JavaScript
(`simulation.js`) and a line-for-line C++ port (`physics.cpp`) compiled to
WebAssembly. Both implement the same tiny engine interface and are switched live
with **P** or the physics button, *mid-scene* — the bodies carry over, and the
incoming engine adopts their exact state.

The point is not just "C++ is faster." The two are behaviorally identical, so
switching between them makes the performance difference tangible against the
built-in FPS counter. On the machine this was developed on, a ~400-body
avalanche cost **2.65 ms/step in JavaScript** and **0.76 ms/step in
WebAssembly** — about a **3.5× speedup**.

The interesting part is *why*, and it is not raw arithmetic — modern JavaScript
JITs are genuinely fast at math. It is **memory**. The JavaScript solver, in the
course of a heavy step, allocates millions of tiny `{x, y, z}` vector objects,
and the garbage collector has to chase them all. The C++ engine allocates
*nothing* after startup: every body, contact, and hash cell lives in a flat
static array sized at compile time. No GC, better cache locality. That is where
most of the 3.5× comes from.

**How the boundary works** (`physics-wasm.js`): WebAssembly runs in the same
thread as your JavaScript and shares a block of **linear memory** — one big
`ArrayBuffer` that holds all the C++ data. JavaScript calls exported C++
functions (`step`, `add_body`, `bounce_all`, …) like ordinary functions, and
reads results back by creating a `Float64Array` **view** directly over the WASM
memory — zero copy. Each frame, the wrapper steps the C++ solver and then syncs
the packed state buffer back into the shared `bodies` objects (matched by a
stable id), firing the same renderer cache hooks the JS engine fires. Because the
*same* body objects flow between both engines, switching engines with a scene
mid-flight just works.

A note on determinism: the two engines are behaviorally equivalent but not
bit-identical. C++ and the JS JIT make slightly different floating-point choices,
so the same drop diverges microscopically over time. Both settle into the same
kinds of piles; neither is "more correct."

---

## The camera

A **fly camera** defined by a position and yaw/pitch. Mouse look and orbit
accumulate in event handlers and are consumed once per frame. Orbit is
implemented as a rotation of the whole camera rig about the floor's central
vertical axis — mathematically identical to spinning the scene, but the physics
world stays put. The projection is a hand-rolled perspective transform; the same
math, run in reverse, turns a mouse click into a world-space ray that is
intersected with the floor to decide where a dropped body lands. The floor and
its grid are clipped against the near plane in 3D so they survive low, close
viewpoints.

---

## Building the WebAssembly module

You only need this if you edit `physics.cpp`. The compiled output is committed,
so running the project needs no tools.

**Prerequisites:** the [Emscripten SDK](https://emscripten.org) (`emsdk`) and
Python on your PATH. Emscripten is a self-contained LLVM/Clang-based toolchain —
it does *not* use GCC, and installs nothing else on your system:

```bat
git clone https://github.com/emscripten-core/emsdk
cd emsdk
emsdk install latest
emsdk activate latest
```

Then, from the project directory:

```bat
build-wasm.bat
```

The script does two things: it compiles `physics.cpp` to a standalone
`physics.wasm` (~24 KB, no JS glue, all static memory), and then embeds that
binary as a base64 string in `physics-wasm-bin.js`. The embedding is what lets
the page keep working from `file://` — browsers refuse to `fetch()` a `.wasm`
file from disk, but a base64 string in a `.js` file loads fine anywhere. Adjust
the `EMCC` path at the top of the script if your `emsdk` lives somewhere other
than `D:\WebDev\emsdk`.

If you deploy to a web server and don't care about `file://`, you can delete the
base64 step and have `physics-wasm.js` `fetch('physics.wasm')` directly — a
few-line change that trades the ~33% base64 size overhead for a `.wasm` MIME-type
config on the server.

---

## Performance notes

- **Settled scenes are nearly free.** Sleeping means a thousand resting bodies
  cost almost nothing to simulate, and the software renderer's cache means they
  cost almost nothing to draw. The load is entirely in the *awake* bodies.
- **Rendering is never the bottleneck on a GPU.** The scene is ~12k triangles —
  a rounding error for any GPU of the last 15 years, whether via WebGL or
  WebGPU. The two GPU renderers perform essentially identically here; WebGPU's
  advantages (lower CPU overhead across thousands of draw calls, compute) only
  show up at workloads far larger than this.
- **The CPU limit is the physics, and it scales with awake bodies.** As a rough
  rule of thumb, expect a few hundred simultaneously-awake bodies at 60 fps on
  the JavaScript engine, and a couple of thousand on the WebAssembly engine,
  before the fixed-timestep loop starts gracefully slowing the simulation. The
  **B** key (which wakes everything at once) is the heaviest thing you can do and
  makes a good stress test.
- The software renderer is **pixel-bound** — its cost scales with resolution, so
  it is the one renderer that slows down at fullscreen. The GPU renderers are
  resolution-independent for a scene this small.

The project was deliberately kept **single-threaded and readable** rather than
optimized. The obvious next levers — allocation-free vector math in the JS hot
path, contact warm-starting, multi-threaded island solving, GPU compute — would
each cost clarity, which is the one thing this codebase is trying to preserve.

---

## Credits and license

The code was written by **Claude Fable 5** (Anthropic).

No third-party code, assets, or libraries are used anywhere in the project.

Released under the [MIT License](LICENSE). Use it, learn from it, take it apart.
