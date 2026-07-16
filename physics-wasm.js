"use strict";
// Fable 5 Physics Sandbox — written by Claude Fable 5 (Anthropic).
// MIT License; see LICENSE and README.md.
// ===========================================================================
// WASM PHYSICS ENGINE — JavaScript wrapper around physics.cpp.
//
// The compiled module (embedded as base64 in physics-wasm-bin.js so the page
// still works from file://) owns the authoritative simulation state in its
// linear memory. This wrapper implements the same engine interface as the
// JavaScript engine in simulation.js:
//
//   - spawn/remove/bounce/clear forward to exported C++ functions;
//   - step() runs the C++ solver, then SYNCS the results back into the
//     shared `bodies` array by viewing the WASM memory directly as a
//     Float64Array (zero copy across the boundary) — so every renderer
//     keeps working untouched, reading the same body objects as always;
//   - sleep/wake transitions and removals detected during the sync fire the
//     same renderer cache hooks the JavaScript engine fires.
//
// Bodies are matched across the boundary by a stable id this wrapper assigns
// at spawn. Body objects in `bodies` are never recreated — the same objects
// flow between both engines, which is what makes live engine switching
// (with the scene mid-flight) possible.
// ===========================================================================

const WasmPhysics = (() => {
    let ex = null;                  // wasm exports
    const KINDS = { cube: 0, sphere: 1, tet: 2 };
    let nextId = 1;

    function addToWasm(b) {
        b.wasmId = nextId++;
        ex.add_body(b.wasmId, KINDS[b.shape.name],
            b.pos.x, b.pos.y, b.pos.z,
            b.q.w, b.q.x, b.q.y, b.q.z,
            b.vel.x, b.vel.y, b.vel.z,
            b.angVel.x, b.angVel.y, b.angVel.z,
            b.sleeping ? 1 : 0);
    }

    // Pull the packed state (16 doubles per body — see physics.cpp) back
    // into the shared body objects, firing renderer hooks on transitions.
    function sync() {
        const n = ex.body_count();
        const f = new Float64Array(ex.memory.buffer, ex.state_ptr(), n * 16);
        const seen = new Set();
        const byId = new Map();
        for (const b of bodies) byId.set(b.wasmId, b);
        for (let i = 0; i < n; i++) {
            const o = i * 16;
            const b = byId.get(f[o]);
            if (!b) continue;
            seen.add(f[o]);
            const wasAsleep = b.sleeping;
            const sleeping = f[o + 15] > 0.5;
            b.pos.x = f[o + 2]; b.pos.y = f[o + 3]; b.pos.z = f[o + 4];
            b.q.w = f[o + 5]; b.q.x = f[o + 6]; b.q.y = f[o + 7]; b.q.z = f[o + 8];
            b.vel.x = f[o + 9]; b.vel.y = f[o + 10]; b.vel.z = f[o + 11];
            b.angVel.x = f[o + 12]; b.angVel.y = f[o + 13]; b.angVel.z = f[o + 14];
            if (!sleeping || wasAsleep !== sleeping) b.axes = quatAxes(b.q);
            if (wasAsleep && !sleeping && renderer) renderer.bodyWoke(b);
            b.sleeping = sleeping;
        }
        // ids gone from the wasm side were removed (fell below the world)
        for (let i = bodies.length - 1; i >= 0; i--) {
            if (!seen.has(bodies[i].wasmId)) {
                if (renderer) renderer.bodyRemoved(bodies[i]);
                bodies.splice(i, 1);
            }
        }
    }

    return {
        name: 'WebAssembly',

        async init() {
            if (ex) return true;
            if (typeof PHYSICS_WASM_BASE64 === 'undefined' || typeof WebAssembly === 'undefined') return false;
            const bytes = Uint8Array.from(atob(PHYSICS_WASM_BASE64), ch => ch.charCodeAt(0));
            const module = await WebAssembly.compile(bytes);
            // standalone WASM may declare imports (e.g. wasi exit): stub them
            const imports = {};
            for (const im of WebAssembly.Module.imports(module)) {
                (imports[im.module] = imports[im.module] || {})[im.name] = () => {};
            }
            const instance = await WebAssembly.instantiate(module, imports);
            ex = instance.exports;
            if (ex._initialize) ex._initialize();   // wasi reactor init
            ex.init();
            return true;
        },

        // take over the scene from the other engine, wherever it left off
        activate() {
            ex.clear_scene();
            for (const b of bodies) addToWasm(b);
        },

        step(h) {
            ex.step(h);
            sync();
        },

        bodyAdded(b) { addToWasm(b); },
        bodyDropped(gone) { ex.remove_body(gone.wasmId); },   // wakes its neighbors
        bounce() { ex.bounce_all(Math.random() * 2147483647); },
        clear() { ex.clear_scene(); }
    };
})();
