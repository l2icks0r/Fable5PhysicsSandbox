"use strict";
// Fable 5 Physics Sandbox — written by Claude Fable 5 (Anthropic).
// MIT License; see LICENSE and README.md.
// ===========================================================================
// SOFTWARE RENDERER — everything is drawn by the CPU.
//
// Cubes are rasterized scanline-by-scanline into an ImageData pixel buffer
// with a per-pixel z-buffer (1/z is affine across a projected flat face, so
// its gradient is computed once per face and walked incrementally). The
// antialiased floor, grid, and shadows are drawn with the 2D canvas API and
// the cube layer is composited on top.
//
// Because CPU pixel-filling is expensive, this renderer works hard to avoid
// it: everything that isn't moving — the floor and every sleeping cube — is
// baked into cached color+depth layers, and only moving cubes are rasterized
// per frame. Wakes and removals invalidate just the dirty screen rectangle
// around them (the bodyWoke / bodyRemoved / sceneCleared hooks below). While
// the camera moves the cache is useless, so the whole scene is redrawn
// directly until it comes to rest.
//
// Compare with renderer-webgl.js / renderer-webgpu.js: the GPU makes all of
// this caching machinery unnecessary — those renderers simply redraw the
// whole world every frame.
// ===========================================================================

const SoftwareRenderer = (() => {
    let canvas, ctx;                                   // visible canvas
    let cubeLayer, cubeCtx, cubeImage, buf32, zbuf;    // per-frame raster target
    let bgLayer, bgCtx, staticBuf32, staticZbuf;       // cached: floor + sleeping cubes
    let rBuf, rZ;                                      // rasterPoly's current target
    let scX0 = 0, scY0 = 0, scX1 = 0, scY1 = 0;        // rasterPoly scissor bounds
    let cubeLayerClean = false;                        // cube layer already matches the cache
    let staticDirty = true;                            // static layers need a rebuild
    let moving = false;                                // camera moved last frame
    // dirty screen rect: the only region of the static layers needing a rebuild
    let dirtyX0 = Infinity, dirtyY0 = Infinity, dirtyX1 = -Infinity, dirtyY1 = -Infinity;

    // ======================= pixel-level rasterizer =====================

    function shadeInt(base, n) {   // flat Lambert shade packed as canvas ABGR
        const m = AMBIENT + DIFFUSE * Math.max(0, dot(n, LIGHT_DIR));
        const r = Math.min(255, base.r * m | 0);
        const g = Math.min(255, base.g * m | 0);
        const b = Math.min(255, base.b * m | 0);
        return (255 << 24) | (b << 16) | (g << 8) | r;
    }

    // Scanline fill of a convex planar screen polygon with per-pixel z-testing.
    // A flat 3D face projects so that 1/z is affine in screen space, so its
    // gradient is computed once per face and walked incrementally along each
    // span; only actually covered pixels are touched.
    function rasterPoly(pts, col) {
        const n = pts.length;
        const p0 = pts[0];
        const x0 = p0.x, y0 = p0.y, z0 = p0.iz;
        // depth gradient from the first non-degenerate vertex triple
        let izX = 0, izY = 0, ok = false;
        for (let k = 2; k < n && !ok; k++) {
            const dx1 = pts[k - 1].x - x0, dy1 = pts[k - 1].y - y0, dz1 = pts[k - 1].iz - z0;
            const dx2 = pts[k].x - x0, dy2 = pts[k].y - y0, dz2 = pts[k].iz - z0;
            const det = dx1 * dy2 - dy1 * dx2;
            if (Math.abs(det) > 1e-6) {
                izX = (dz1 * dy2 - dy1 * dz2) / det;   // d(1/z)/dx
                izY = (dx1 * dz2 - dz1 * dx2) / det;   // d(1/z)/dy
                ok = true;
            }
        }
        if (!ok) return;

        let minY = Infinity, maxY = -Infinity;
        for (const p of pts) {
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        const yStart = Math.max(scY0, Math.ceil(minY - 0.5));
        const yEnd = Math.min(scY1, Math.floor(maxY - 0.5));

        for (let y = yStart; y <= yEnd; y++) {
            const yc = y + 0.5;
            let xL = Infinity, xR = -Infinity;
            for (let k = 0; k < n; k++) {   // span = where the scanline crosses the edges
                const a = pts[k], b = pts[(k + 1) % n];
                if ((a.y <= yc) === (b.y <= yc)) continue;
                const x = a.x + (yc - a.y) * (b.x - a.x) / (b.y - a.y);
                if (x < xL) xL = x;
                if (x > xR) xR = x;
            }
            const xStart = Math.max(scX0, Math.ceil(xL - 0.5));
            const xEnd = Math.min(scX1, Math.ceil(xR - 0.5) - 1);
            let iz = z0 + (xStart + 0.5 - x0) * izX + (yc - y0) * izY;
            let idx = y * W + xStart;
            for (let x = xStart; x <= xEnd; x++, idx++, iz += izX) {
                if (iz > rZ[idx]) {
                    rZ[idx] = iz;
                    rBuf[idx] = col;
                }
            }
        }
    }

    // Depth-only pass: write the finite floor quad into the z-buffer with a
    // transparent color, so anything below the plane is depth-rejected and the
    // antialiased canvas floor shows through instead. The slight depth bias
    // keeps resting cubes (which sink up to the contact slop) winning ties.
    function rasterFloorDepth() {
        const nearD = dot(camPos, camF) + 0.15;
        let poly = [
            v3(-FLOOR_HALF, 0, -FLOOR_HALF), v3(FLOOR_HALF, 0, -FLOOR_HALF),
            v3(FLOOR_HALF, 0, FLOOR_HALF), v3(-FLOOR_HALF, 0, FLOOR_HALF)
        ];
        poly = clipPoly(poly, neg(camF), -nearD);   // may become a pentagon
        if (poly.length < 3) return;
        const pts = [];
        for (const v of poly) {
            const p = project(v);
            if (!p) return;
            pts.push({ x: p.x, y: p.y, iz: p.iz * 0.999 });
        }
        rasterPoly(pts, 0);
    }

    // ========================= canvas-level helpers =====================

    function pathPoly(g, pts) {
        g.beginPath();
        g.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) g.lineTo(pts[i].x, pts[i].y);
        g.closePath();
    }

    function hull2(pts) {  // 2D convex hull (x,z) via monotone chain
        pts = pts.slice().sort((p, q) => p.x - q.x || p.z - q.z);
        const cr = (o, a, b) => (a.x - o.x) * (b.z - o.z) - (a.z - o.z) * (b.x - o.x);
        const lo = [], hi = [];
        for (const p of pts) {
            while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop();
            lo.push(p);
        }
        for (let i = pts.length - 1; i >= 0; i--) {
            const p = pts[i];
            while (hi.length >= 2 && cr(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop();
            hi.push(p);
        }
        lo.pop(); hi.pop();
        return lo.concat(hi);
    }

    // Black backdrop, grey floor plane, and grid, drawn with the current
    // camera. The floor quad and grid lines are clipped against the near
    // plane in 3D before projecting, so they survive low/close views.
    function drawFloor(g) {
        g.fillStyle = '#000';
        g.fillRect(0, 0, W, H);
        const NEAR = 0.15;
        const nearD = dot(camPos, camF) + NEAR;
        let poly = [
            v3(-FLOOR_HALF, 0, -FLOOR_HALF), v3(FLOOR_HALF, 0, -FLOOR_HALF),
            v3(FLOOR_HALF, 0, FLOOR_HALF), v3(-FLOOR_HALF, 0, FLOOR_HALF)
        ];
        poly = clipPoly(poly, neg(camF), -nearD);   // keep dot(p, camF) >= nearD
        if (poly.length < 3) return;
        const pts = poly.map(project);
        if (pts.some(p => !p)) return;
        pathPoly(g, pts);
        g.fillStyle = '#7d7d7d';
        g.fill();
        g.strokeStyle = 'rgba(0,0,0,0.15)';
        g.lineWidth = 1;
        const line = (a, b) => {
            const za = dot(sub(a, camPos), camF), zb = dot(sub(b, camPos), camF);
            if (za < NEAR && zb < NEAR) return;
            if (za < NEAR) a = add(a, mul(sub(b, a), (NEAR - za) / (zb - za)));
            else if (zb < NEAR) b = add(b, mul(sub(a, b), (NEAR - zb) / (za - zb)));
            const p = project(a), q = project(b);
            if (!p || !q) return;
            g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(q.x, q.y); g.stroke();
        };
        for (let gl = -FLOOR_HALF; gl <= FLOOR_HALF; gl += 2) {
            line(v3(gl, 0, -FLOOR_HALF), v3(gl, 0, FLOOR_HALF));
            line(v3(-FLOOR_HALF, 0, gl), v3(FLOOR_HALF, 0, gl));
        }
    }

    // Soft contact shadow: the body's footprint dropped straight onto the
    // floor, filled with height-faded translucent black. Polyhedra project
    // their vertex hull; a sphere's footprint is simply a circle.
    function drawShadow(g, b) {
        if (b.pos.y < -0.5 || Math.abs(b.pos.x) > FLOOR_HALF || Math.abs(b.pos.z) > FLOOR_HALF) return;
        let hull;
        if (b.shape.kind === 'sphere') {
            hull = [];
            for (let k = 0; k < 16; k++) {
                const a = (k / 16) * 2 * Math.PI;
                hull.push(project(v3(b.pos.x + Math.cos(a) * b.shape.r, 0,
                                     b.pos.z + Math.sin(a) * b.shape.r)));
            }
        } else {
            const flat = worldVerts(b).map(v => ({ x: v.x, z: v.z }));
            hull = hull2(flat).map(p => project(v3(p.x, 0, p.z)));
        }
        if (hull.length < 3 || hull.some(p => !p)) return;
        pathPoly(g, hull);
        g.fillStyle = 'rgba(0,0,0,' + (0.35 * Math.max(0.08, 1 - b.pos.y / 12)).toFixed(3) + ')';
        g.fill();
    }

    // Rasterize a convex polyhedron's visible faces into rBuf/rZ.
    function drawConvex(b) {
        const verts = worldVerts(b);
        const proj = verts.map(project);
        for (const f of b.shape.faces) {
            const n = rotateLocal(b, f.n);
            if (dot(n, sub(camPos, verts[f.v[0]])) <= 0) continue;   // backface
            const pts = [];
            let ok = true;
            for (const i of f.v) {
                if (!proj[i]) { ok = false; break; }
                pts.push(proj[i]);
            }
            if (ok) rasterPoly(pts, shadeInt(b.color, n));
        }
    }

    // Spheres are drawn analytically: for each pixel in the projected
    // bounding box, intersect the eye ray with the sphere and shade the hit
    // with a per-pixel normal — an exact silhouette and smooth lighting with
    // no mesh at all (the GPU renderers approximate the same ball with
    // triangles). The beach-ball wedges come from the hit point expressed in
    // the sphere's own rotating frame, so the pattern tumbles with the body.
    function drawSphere(b) {
        const r = b.shape.r;
        const oc = sub(b.pos, camPos);                 // eye -> center
        const zc = dot(oc, camF);
        if (zc <= 0.15) return;                        // behind the near limit
        // A perspective-projected sphere is NOT a circle around its projected
        // center: off-axis it becomes an ellipse that bulges away from the
        // screen center. Bound each screen axis exactly with the eye rays
        // tangent to the sphere in that axis' plane: the tangent slopes s
        // solve s^2(zc^2 - r^2) - 2*s*c*zc + c^2 - r^2 = 0 for a center at
        // (c, zc) in that plane.
        const cx = dot(oc, camR), cy = dot(oc, camU);
        const axisBounds = (c) => {
            const inv = zc * zc - r * r;
            const disc = c * c + inv;
            if (inv <= 0) return null;                 // sphere too close: no tangents
            const sq = r * Math.sqrt(disc);
            return [(c * zc - sq) / inv, (c * zc + sq) / inv];
        };
        const bx = axisBounds(cx), by = axisBounds(cy);
        const x0 = Math.max(scX0, bx ? Math.floor(W / 2 + bx[0] * projScale) : scX0);
        const x1 = Math.min(scX1, bx ? Math.ceil(W / 2 + bx[1] * projScale) : scX1);
        const y0 = Math.max(scY0, by ? Math.floor(H / 2 - by[1] * projScale) : scY0);
        const y1 = Math.min(scY1, by ? Math.ceil(H / 2 - by[0] * projScale) : scY1);
        const c2 = len2(oc) - r * r;
        const white = { r: 242, g: 242, b: 242 };
        for (let y = y0; y <= y1; y++) {
            const ay = (H / 2 - (y + 0.5)) / projScale;
            const rowDir = add(camF, mul(camU, ay));   // camF component stays 1
            for (let x = x0; x <= x1; x++) {
                const ax = (x + 0.5 - W / 2) / projScale;
                const dir = add(rowDir, mul(camR, ax));
                const bq = dot(dir, oc);
                const disc = bq * bq - len2(dir) * c2;
                if (disc <= 0) continue;               // ray misses the sphere
                const t = (bq - Math.sqrt(disc)) / len2(dir);
                if (t <= 0.15) continue;
                const iz = 1 / t;                      // view depth = t (dir·camF = 1)
                const idx = y * W + x;
                if (iz <= rZ[idx]) continue;
                const n = mul(sub(add(camPos, mul(dir, t)), b.pos), 1 / r);
                const local = toLocal(b, n);
                const wedge = Math.floor((Math.atan2(local.z, local.x) / (2 * Math.PI) + 0.5) * 6);
                const base = wedge % 2 === 1 ? white : b.color;
                rZ[idx] = iz;
                rBuf[idx] = shadeInt(base, n);
            }
        }
    }

    function drawBody(b) {
        if (b.shape.kind === 'sphere') drawSphere(b);
        else drawConvex(b);
    }

    // Front-to-back lets occluded pixels fail the depth test without writes.
    const frontToBack = (a, b) => len2(sub(a.pos, camPos)) - len2(sub(b.pos, camPos));

    // ===================== static-layer cache machinery =================

    // Exact screen-space footprint of a body: the bounding box of its 8
    // projected corners plus their floor projections (which bound the shadow
    // hull), padded for antialiasing. It is stored on the body while it
    // sleeps — sleepers don't move, so it stays valid until the body wakes.
    // Any corner behind the near plane falls back to the whole screen.
    function screenRectOf(b) {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const v of worldVerts(b)) {
            const p = project(v);
            const s = project(v3(v.x, 0, v.z));
            if (!p || !s) return { x0: 0, y0: 0, x1: W, y1: H };
            x0 = Math.min(x0, p.x, s.x);
            y0 = Math.min(y0, p.y, s.y);
            x1 = Math.max(x1, p.x, s.x);
            y1 = Math.max(y1, p.y, s.y);
        }
        return { x0: x0 - 3, y0: y0 - 3, x1: x1 + 3, y1: y1 + 3 };
    }

    function markDirtyRect(b) {
        const r = b.screenRect || (b.screenRect = screenRectOf(b));
        dirtyX0 = Math.min(dirtyX0, r.x0);
        dirtyY0 = Math.min(dirtyY0, r.y0);
        dirtyX1 = Math.max(dirtyX1, r.x1);
        dirtyY1 = Math.max(dirtyY1, r.y1);
        staticDirty = true;
    }

    // Bake everything that isn't moving — background, floor, grid, sleeping
    // cubes and their shadows — into reusable layers. Wakes and removals are
    // localized, so only the dirty screen rect is rebuilt: it is cleared,
    // and just the sleepers whose footprint touches it are re-rasterized
    // (scissored), leaving the rest of the cache untouched.
    function buildStaticLayers() {
        const x0 = Math.max(0, Math.floor(dirtyX0)), y0 = Math.max(0, Math.floor(dirtyY0));
        const x1 = Math.min(W - 1, Math.ceil(dirtyX1)), y1 = Math.min(H - 1, Math.ceil(dirtyY1));
        dirtyX0 = dirtyY0 = Infinity;
        dirtyX1 = dirtyY1 = -Infinity;
        staticDirty = false;
        cubeLayerClean = false;
        if (x0 > x1 || y0 > y1) return;

        bgCtx.save();
        bgCtx.beginPath();
        bgCtx.rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        bgCtx.clip();
        drawFloor(bgCtx);

        for (let y = y0; y <= y1; y++) {   // clear depth + color inside the rect
            staticBuf32.fill(0, y * W + x0, y * W + x1 + 1);
            staticZbuf.fill(0, y * W + x0, y * W + x1 + 1);
        }
        rBuf = staticBuf32;
        rZ = staticZbuf;
        scX0 = x0; scY0 = y0; scX1 = x1; scY1 = y1;
        rasterFloorDepth();   // the finite floor occludes anything below it

        // Cached sleepers redraw whenever they overlap the rect (their pixels
        // outside it are already baked). An uncached sleeper may only be baked
        // when the rect covers its whole on-screen footprint — a partial,
        // scissored bake marked as cached would leave permanent holes.
        const shouldDraw = (b) => {
            if (!b.sleeping) return false;
            const r = b.screenRect || (b.screenRect = screenRectOf(b));
            if (b.cached) return r.x1 >= x0 && r.x0 <= x1 && r.y1 >= y0 && r.y0 <= y1;
            return Math.max(r.x0, 0) >= x0 && Math.min(r.x1, W - 1) <= x1 &&
                   Math.max(r.y0, 0) >= y0 && Math.min(r.y1, H - 1) <= y1;
        };
        const sleepers = bodies.filter(shouldDraw).sort(frontToBack);
        for (const b of sleepers) {
            drawShadow(bgCtx, b);
            drawBody(b);
            b.cached = true;
        }
        scX0 = 0; scY0 = 0; scX1 = W - 1; scY1 = H - 1;
        bgCtx.restore();
    }

    // ============================ frame paths ===========================

    // Camera in motion: the cached layers are from the old viewpoint, so
    // draw the whole world directly. This is the simple, brute-force path —
    // and exactly what the GPU renderers do every frame.
    function drawDirect() {
        drawFloor(ctx);
        const all = bodies.slice().sort(frontToBack);
        buf32.fill(0);
        zbuf.fill(0);
        rBuf = buf32;
        rZ = zbuf;
        rasterFloorDepth();
        for (const b of all) drawShadow(ctx, b);
        for (const b of all) drawBody(b);
        cubeCtx.putImageData(cubeImage, 0, 0);
        ctx.drawImage(cubeLayer, 0, 0);
    }

    // Camera at rest: composite the cached layers and rasterize only the
    // cubes that are moving (or not yet baked).
    function drawCached() {
        let awake = 0, pendingCache = 0;
        for (const b of bodies) {
            if (!b.sleeping) awake++;
            else if (!b.cached) pendingCache++;
        }
        // rebuild layers on wake/removal, or once the whole scene has settled
        if (awake === 0 && pendingCache > 0) {
            for (const b of bodies) if (b.sleeping && !b.cached) markDirtyRect(b);
        }
        if (staticDirty) buildStaticLayers();

        ctx.drawImage(bgLayer, 0, 0);

        // moving (and not-yet-cached) cubes: shadows onto the canvas, bodies
        // z-buffered on top of the cached depth image so they occlude and are
        // occluded by the sleeping pile correctly
        const dyn = bodies.filter(b => !b.cached).sort(frontToBack);
        if (dyn.length || !cubeLayerClean) {
            buf32.set(staticBuf32);
            zbuf.set(staticZbuf);
            rBuf = buf32;
            rZ = zbuf;
            for (const b of dyn) drawBody(b);
            cubeCtx.putImageData(cubeImage, 0, 0);
            cubeLayerClean = dyn.length === 0;
        }
        for (const b of dyn) drawShadow(ctx, b);
        ctx.drawImage(cubeLayer, 0, 0);
    }

    // ========================= renderer interface =======================

    return {
        name: 'software',

        init(cv) {
            canvas = cv;
            ctx = cv.getContext('2d');
            this.resize();
            return true;
        },

        resize() {
            cubeLayer = document.createElement('canvas');
            cubeLayer.width = W;
            cubeLayer.height = H;
            cubeCtx = cubeLayer.getContext('2d');
            cubeImage = new ImageData(W, H);
            buf32 = new Uint32Array(cubeImage.data.buffer);
            zbuf = new Float32Array(W * H);
            bgLayer = document.createElement('canvas');
            bgLayer.width = W;
            bgLayer.height = H;
            bgCtx = bgLayer.getContext('2d');
            staticBuf32 = new Uint32Array(W * H);
            staticZbuf = new Float32Array(W * H);
            scX0 = 0; scY0 = 0; scX1 = W - 1; scY1 = H - 1;
            for (const b of bodies) {
                b.cached = false;
                b.screenRect = null;   // stored footprints are in the old projection
            }
            dirtyX0 = 0; dirtyY0 = 0; dirtyX1 = W; dirtyY1 = H;
            staticDirty = true;
            cubeLayerClean = false;
        },

        dispose() {},

        render(camMoved) {
            if (camMoved) {
                for (const b of bodies) {   // every baked footprint is stale
                    b.cached = false;
                    b.screenRect = null;
                }
                moving = true;
                drawDirect();
                return;
            }
            if (moving) {
                moving = false;   // camera came to rest: rebake everything once
                dirtyX0 = 0; dirtyY0 = 0; dirtyX1 = W; dirtyY1 = H;
                staticDirty = true;
                cubeLayerClean = false;
            }
            drawCached();
        },

        // cache hooks: the physics engine reports events that make baked
        // pixels stale (the GPU renderers implement these as no-ops)
        bodyWoke(b) {
            if (b.cached) { markDirtyRect(b); b.cached = false; }
            b.screenRect = null;   // it will move, so its footprint is stale too
        },
        bodyRemoved(b) {
            if (b.cached) { markDirtyRect(b); b.cached = false; }
        },
        sceneCleared() {
            dirtyX0 = 0; dirtyY0 = 0; dirtyX1 = W; dirtyY1 = H;
            staticDirty = true;
            cubeLayerClean = false;
        }
    };
})();
