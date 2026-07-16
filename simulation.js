"use strict";
// Fable 5 Physics Sandbox — created by Claude Fable 5 (Anthropic), directed by
// the repository author. MIT License; see LICENSE and README.md.
// ===========================================================================
// SIMULATION — physics, camera, input, and the main loop.
//
// This file knows nothing about how pixels get drawn. Rendering is delegated
// to whichever renderer is active (see the registry near the bottom); the
// three implementations live in renderer-software.js, renderer-webgl.js and
// renderer-webgpu.js, all behind the same small interface:
//
//   name              label shown on the gfx button
//   init(canvas)      set up; may be async; false/throw = unsupported
//   render(camMoved)  draw the current world state
//   resize()          the canvas was resized
//   dispose()         release the context (renderer is being switched out)
//   bodyWoke(b), bodyRemoved(b), sceneCleared()
//                     cache-invalidation hooks; only the caching software
//                     renderer does anything with them
// ===========================================================================

// ============================== vec3 ==============================
const v3    = (x, y, z) => ({ x, y, z });
const add   = (a, b) => v3(a.x + b.x, a.y + b.y, a.z + b.z);
const sub   = (a, b) => v3(a.x - b.x, a.y - b.y, a.z - b.z);
const mul   = (a, s) => v3(a.x * s, a.y * s, a.z * s);
const neg   = (a)    => v3(-a.x, -a.y, -a.z);
const dot   = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a, b) => v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const len2  = (a)    => dot(a, a);
const norm  = (a)    => { const l = Math.sqrt(len2(a)) || 1; return mul(a, 1 / l); };
const clampN = (x, lo, hi) => x < lo ? lo : (x > hi ? hi : x);

// ============================ quaternion ===========================
function qMul(a, b) {
    return {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}
function qNormalize(q) {
    const l = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) || 1;
    q.w /= l; q.x /= l; q.y /= l; q.z /= l;
    return q;
}
// Columns of the rotation matrix = the body's local axes in world space.
function quatAxes(q) {
    const { w, x, y, z } = q;
    return [
        v3(1 - 2 * (y * y + z * z), 2 * (x * y + z * w),     2 * (x * z - y * w)),
        v3(2 * (x * y - z * w),     1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
        v3(2 * (x * z + y * w),     2 * (y * z - x * w),     1 - 2 * (x * x + y * y))
    ];
}
function randomQuat() {   // uniformly random unit quaternion (Shoemake)
    const u1 = Math.random(), u2 = Math.random() * 2 * Math.PI, u3 = Math.random() * 2 * Math.PI;
    const s1 = Math.sqrt(1 - u1), s2 = Math.sqrt(u1);
    return { w: s1 * Math.sin(u2), x: s1 * Math.cos(u2), y: s2 * Math.sin(u3), z: s2 * Math.cos(u3) };
}
// Integrate orientation in place: q += 0.5 * (w as pure quaternion) * q * h.
function qIntegrate(q, w, h) {
    const dq = qMul({ w: 0, x: w.x, y: w.y, z: w.z }, q);
    q.w += 0.5 * dq.w * h;
    q.x += 0.5 * dq.x * h;
    q.y += 0.5 * dq.y * h;
    q.z += 0.5 * dq.z * h;
    return qNormalize(q);
}

// ============================ constants ============================
const GRAVITY        = -9.81;
const CUBE_HALF      = 0.5;
const CUBE_MASS      = 1.0;
const CUBE_RADIUS    = CUBE_HALF * Math.sqrt(3);   // bounding sphere
const FLOOR_HALF     = 12;
const SPAWN_HEIGHT   = 6.5;
const MAX_CUBES      = 1000;

// Materials: plastic die on a wood plane -> low elasticity, moderate friction.
const E_FLOOR   = 0.22;   // restitution, plastic on wood
const E_CUBE    = 0.12;   // restitution, plastic on plastic
const MU_FLOOR  = 0.45;   // friction, plastic on wood
const MU_CUBE   = 0.35;   // friction, plastic on plastic

const FIXED_DT       = 1 / 120;
const SOLVER_ITERS   = 10;
const BAUMGARTE      = 0.2;
const MAX_BIAS       = 4.0;   // cap correction speed so deep spawn overlaps don't explode
const PEN_SLOP       = 0.01;
const REST_THRESHOLD = 0.8;   // approach speed below which bounces are killed
const LIN_DAMP       = 0.01;
const ANG_DAMP       = 0.05;
const ROLL_RESIST    = 1.2;   // extra spin damping for touching spheres: point-contact
                              // friction can't slow pure rolling (contact velocity is
                              // zero), so without this a ball would roll forever

// Sleeping: a cube in contact that stays this slow for this long freezes
// until something hits or shoves it, so settled piles cost almost nothing.
const SLEEP_LIN2   = 0.02;    // linear speed^2 threshold
const SLEEP_ANG2   = 0.04;    // angular speed^2 threshold
const SLEEP_STEPS  = 60;      // 0.5 s at the fixed timestep
const WAKE_SPEED2  = 1.0;     // an approaching body faster than 1 u/s wakes a sleeper
const WAKE_IMPULSE = 1.0;     // solved impulse that counts as a shove; static weight
                              // transfer even under a tall stack stays well below this

// Lighting, shared by every renderer so all three look identical.
const LIGHT_DIR = norm(v3(-0.45, 0.85, 0.35));
const AMBIENT = 0.32, DIFFUSE = 0.68;
const UP = v3(0, 1, 0);

// ======================== shape descriptors ========================
// Every body carries a shape descriptor. Convex polyhedra (cube, tet) are
// described by local-space vertices and face loops; makeConvexShape derives
// what the engine needs: outward face normals (winding auto-corrected so
// loops read counter-clockwise from outside), unique edges, deduplicated
// edge directions for SAT axes, and a bounding radius. Spheres are a
// special case handled analytically everywhere.

// Rotate a local-space vector into world space using a body's axes, and back.
const rotateLocal = (b, v) => add(mul(b.axes[0], v.x), add(mul(b.axes[1], v.y), mul(b.axes[2], v.z)));
const toLocal = (b, v) => v3(dot(v, b.axes[0]), dot(v, b.axes[1]), dot(v, b.axes[2]));

function makeConvexShape(name, verts, faceLoops, invI) {
    const faces = faceLoops.map(loop => {
        let n = norm(cross(sub(verts[loop[1]], verts[loop[0]]), sub(verts[loop[2]], verts[loop[0]])));
        if (dot(n, verts[loop[0]]) < 0) {   // shapes are origin-centered: flip inward windings
            loop = loop.slice().reverse();
            n = neg(n);
        }
        return { v: loop, n };
    });
    const edges = [];        // unique vertex-index pairs (floor rim contacts)
    const edgeDirs = [];     // unique directions (SAT cross axes)
    const seen = new Set();
    for (const f of faces) {
        for (let k = 0; k < f.v.length; k++) {
            const i = f.v[k], j = f.v[(k + 1) % f.v.length];
            const key = Math.min(i, j) + '_' + Math.max(i, j);
            if (seen.has(key)) continue;
            seen.add(key);
            edges.push({ i, j });
            const dir = norm(sub(verts[j], verts[i]));
            if (!edgeDirs.some(d => len2(cross(d, dir)) < 1e-6)) edgeDirs.push(dir);
        }
    }
    let radius = 0;
    for (const v of verts) radius = Math.max(radius, Math.sqrt(len2(v)));
    return { name, kind: 'convex', verts, faces, edges, edgeDirs, radius, invI };
}

const TET_S = 0.55;      // tetrahedron vertex scale (edge length = 2*sqrt(2)*TET_S)
const SPHERE_R = 0.5;

const cubeVerts = [];    // vertex i: bit0 -> +x, bit1 -> +y, bit2 -> +z
for (let i = 0; i < 8; i++) {
    cubeVerts.push(v3((i & 1) ? CUBE_HALF : -CUBE_HALF,
                      (i & 2) ? CUBE_HALF : -CUBE_HALF,
                      (i & 4) ? CUBE_HALF : -CUBE_HALF));
}

// A regular tetrahedron is a Platonic solid, so like the cube and sphere its
// inertia tensor is isotropic — the whole solver keeps its scalar-inertia
// simplification. (A square pyramid would not have this property!)
const tetVerts = [
    v3(TET_S, TET_S, TET_S), v3(TET_S, -TET_S, -TET_S),
    v3(-TET_S, TET_S, -TET_S), v3(-TET_S, -TET_S, TET_S)
];

const SHAPES = {
    // solid cube: I = m*(edge^2)/6
    cube: makeConvexShape('cube', cubeVerts,
        [[1, 3, 7, 5], [0, 2, 6, 4], [2, 3, 7, 6], [0, 1, 5, 4], [4, 5, 7, 6], [0, 1, 3, 2]],
        6 / (CUBE_MASS * (2 * CUBE_HALF) * (2 * CUBE_HALF))),
    // regular tetrahedron: I = m*(edge^2)/20, edge^2 = 8*TET_S^2
    tet: makeConvexShape('tet', tetVerts,
        [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        2.5 / (CUBE_MASS * TET_S * TET_S)),
    // solid sphere: I = (2/5)*m*r^2
    sphere: {
        name: 'sphere', kind: 'sphere',
        r: SPHERE_R, radius: SPHERE_R,
        invI: 2.5 / (CUBE_MASS * SPHERE_R * SPHERE_R)
    }
};

// ============================== bodies =============================
const bodies = [];

function hslToRgb(h, s, l) {
    const f = n => {
        const k = (n + h * 12) % 12;
        const a = s * Math.min(l, 1 - l);
        return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    };
    return { r: Math.round(f(0) * 255), g: Math.round(f(8) * 255), b: Math.round(f(4) * 255) };
}

let spawnShape = 'cube';   // which shape the next drop creates (see input section)

function spawnBody(x, z) {
    const q = randomQuat();
    const spin = () => (Math.random() * 2 - 1) * 7;
    const shape = SHAPES[spawnShape];
    const b = {
        shape,
        pos: v3(x, SPAWN_HEIGHT, z),
        vel: v3(0, 0, 0),
        q,
        axes: quatAxes(q),
        angVel: v3(spin(), spin(), spin()),
        invMass: 1 / CUBE_MASS,
        // All three shapes have isotropic inertia tensors, so the world-space
        // inverse inertia stays a single scalar under any rotation.
        invI: shape.invI,
        radius: shape.radius,
        color: hslToRgb(Math.random(), 0.75, 0.55),
        sleeping: false,
        sleepTimer: 0,
        touch: false
    };
    bodies.push(b);
    physics.bodyAdded(b);
    if (bodies.length > MAX_CUBES) {
        const gone = bodies.shift();
        if (renderer) renderer.bodyRemoved(gone);
        physics.bodyDropped(gone);
    }
}

// A body's vertices in world space. For spheres — which have no vertices —
// return the corners of the bounding box instead; only the footprint and
// bounds code uses this for spheres (collision goes through analytic paths).
function worldVerts(b) {
    if (b.shape.kind === 'sphere') {
        const r = b.shape.r, out = [];
        for (let i = 0; i < 8; i++) {
            out.push(v3(b.pos.x + ((i & 1) ? r : -r),
                        b.pos.y + ((i & 2) ? r : -r),
                        b.pos.z + ((i & 4) ? r : -r)));
        }
        return out;
    }
    return b.shape.verts.map(v => add(b.pos, rotateLocal(b, v)));
}

// ================== broadphase: uniform spatial hash ================
// Cell size = sphere diameter, so any two overlapping bounding spheres
// are in the same or adjacent cells. Rebuilt every step; each pair is
// visited exactly once (j > i, and a body lives in exactly one cell).
const CELL = 2 * CUBE_RADIUS;
const cellKey = (cx, cy, cz) => (cx << 20) | (cy << 10) | cz;

function buildGrid() {
    const grid = new Map();
    for (let i = 0; i < bodies.length; i++) {
        const b = bodies[i];
        b.cx = Math.floor(b.pos.x / CELL) + 512;
        b.cy = Math.floor(b.pos.y / CELL) + 512;
        b.cz = Math.floor(b.pos.z / CELL) + 512;
        const key = cellKey(b.cx, b.cy, b.cz);
        const arr = grid.get(key);
        if (arr) arr.push(i); else grid.set(key, [i]);
    }
    return grid;
}

function forEachNeighbor(grid, b, fn) {
    for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
            for (let dz = -1; dz <= 1; dz++) {
                const arr = grid.get(cellKey(b.cx + dx, b.cy + dy, b.cz + dz));
                if (arr) for (const j of arr) fn(j);
            }
        }
    }
}

function wakeBody(b) {
    if (renderer) renderer.bodyWoke(b);   // its baked pixels are stale
    b.sleeping = false;
    b.sleepTimer = 0;
}

function wakeChain(b, grid) {
    wakeBody(b);
    forEachNeighbor(grid, b, (j) => {   // one level: wake whatever rests against it
        const nb = bodies[j];
        if (nb.sleeping && len2(sub(nb.pos, b.pos)) < (nb.radius + b.radius) ** 2) wakeBody(nb);
    });
}

// ===================== collision: cube vs cube =====================
// Separating Axis Test over the 6 face axes and 9 edge-cross axes,
// then contact manifold via reference-face clipping (face case) or
// closest points between supporting edges (edge case).

function clipPoly(poly, n, d) {  // keep points with dot(p, n) <= d
    const out = [];
    for (let i = 0; i < poly.length; i++) {
        const p = poly[i], q = poly[(i + 1) % poly.length];
        const dp = dot(p, n) - d, dq = dot(q, n) - d;
        if (dp <= 0) out.push(p);
        if (dp * dq < 0) {
            const t = dp / (dp - dq);
            out.push(add(p, mul(sub(q, p), t)));
        }
    }
    return out;
}

function closestSegSeg(p1, q1, p2, q2) {
    const d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
    const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    const c = dot(d1, r), b = dot(d1, d2);
    const denom = a * e - b * b;
    let s = denom > 1e-8 ? clampN((b * f - c * e) / denom, 0, 1) : 0;
    let t = (b * s + f) / e;
    if (t < 0)      { t = 0; s = clampN(-c / a, 0, 1); }
    else if (t > 1) { t = 1; s = clampN((b - c) / a, 0, 1); }
    return [add(p1, mul(d1, s)), add(p2, mul(d2, t))];
}

// General convex-vs-convex collision: SAT over both shapes' face normals
// and the cross products of their unique edge directions, then a contact
// manifold by clipping the incident face against the reference face (face
// case) or closest points between the supporting edges (edge case). This
// one routine handles cube-cube, cube-tet, and tet-tet identically.
function convexConvex(A, B) {
    const vertsA = worldVerts(A), vertsB = worldVerts(B);
    const d = sub(B.pos, A.pos);
    const contact = (p, n, pen) => ({ a: A, b: B, p, n, pen, e: E_CUBE, mu: MU_CUBE });

    // Face axes: for each face, separation = how far the OTHER body's nearest
    // vertex sits in front of the face plane (negative = penetrating). Track
    // the least-penetrating face over both bodies.
    let bestFace = { sep: -Infinity };
    const faceAxes = (body, verts, otherVerts, isA) => {
        for (const f of body.shape.faces) {
            const n = rotateLocal(body, f.n);
            const faceD = dot(n, verts[f.v[0]]);
            let minProj = Infinity;
            for (const v of otherVerts) minProj = Math.min(minProj, dot(n, v));
            const sep = minProj - faceD;
            if (sep > bestFace.sep) bestFace = { sep, n, face: f, ref: body, refVerts: verts, isA };
        }
    };
    faceAxes(A, vertsA, vertsB, true);
    if (bestFace.sep > 0) return [];   // separating plane found
    faceAxes(B, vertsB, vertsA, false);
    if (bestFace.sep > 0) return [];

    // Edge-cross axes, oriented A -> B; separation from the projection intervals.
    let bestEdge = { sep: -Infinity };
    for (const da of A.shape.edgeDirs) {
        const ea = rotateLocal(A, da);
        for (const db of B.shape.edgeDirs) {
            let L = cross(ea, rotateLocal(B, db));
            const l2 = len2(L);
            if (l2 < 1e-8) continue;   // parallel edges are covered by the face axes
            L = mul(L, 1 / Math.sqrt(l2));
            if (dot(L, d) < 0) L = neg(L);
            let maxA = -Infinity, minB = Infinity;
            for (const v of vertsA) maxA = Math.max(maxA, dot(L, v));
            for (const v of vertsB) minB = Math.min(minB, dot(L, v));
            const sep = minB - maxA;
            if (sep > 0) return [];
            if (sep > bestEdge.sep) bestEdge = { sep, L, ea, eb: rotateLocal(B, db) };
        }
    }

    // Prefer face contacts; only take the edge case when clearly shallower.
    if (-bestEdge.sep < -bestFace.sep * 0.95 - 1e-3) {
        // Supporting edge: among each body's edges parallel to the chosen
        // direction, the one reaching furthest along the axis.
        const supportEdge = (body, verts, dirW, sign, L) => {
            let best = null, bestProj = -Infinity;
            for (const e of body.shape.edges) {
                const ev = sub(verts[e.j], verts[e.i]);
                if (len2(cross(ev, dirW)) > 1e-6 * len2(ev)) continue;
                const proj = sign * (dot(L, verts[e.i]) + dot(L, verts[e.j]));
                if (proj > bestProj) { bestProj = proj; best = e; }
            }
            return best;
        };
        const eA = supportEdge(A, vertsA, bestEdge.ea, 1, bestEdge.L);
        const eB = supportEdge(B, vertsB, bestEdge.eb, -1, bestEdge.L);
        const [c1, c2] = closestSegSeg(vertsA[eA.i], vertsA[eA.j], vertsB[eB.i], vertsB[eB.j]);
        return [contact(mul(add(c1, c2), 0.5), bestEdge.L, -bestEdge.sep)];
    }

    const ref = bestFace.ref;
    const inc = ref === A ? B : A;
    const incVerts = ref === A ? vertsB : vertsA;
    const refVerts = bestFace.refVerts;
    const refN = bestFace.n;                     // outward from ref, toward inc
    const n = bestFace.isA ? refN : neg(refN);   // contact normal A -> B

    // Incident face: the face of the other body most anti-parallel to refN.
    let incFace = null, most = Infinity;
    for (const f of inc.shape.faces) {
        const facing = dot(rotateLocal(inc, f.n), refN);
        if (facing < most) { most = facing; incFace = f; }
    }
    let poly = incFace.v.map(i => incVerts[i]);

    // Clip against the side planes of the reference face (loops are CCW from
    // outside, so cross(edge, refN) points out of the face).
    const refLoop = bestFace.face.v;
    for (let k = 0; k < refLoop.length && poly.length; k++) {
        const p = refVerts[refLoop[k]], q = refVerts[refLoop[(k + 1) % refLoop.length]];
        const side = norm(cross(sub(q, p), refN));
        poly = clipPoly(poly, side, dot(side, p));
    }

    // Every clipped point below the reference face becomes a contact.
    const faceD = dot(refN, refVerts[refLoop[0]]);
    const out = [];
    for (const pt of poly) {
        const sep = dot(refN, pt) - faceD;
        if (sep < 0) out.push(contact(pt, n, -sep));
    }
    if (!out.length) out.push(contact(mul(add(A.pos, B.pos), 0.5), n, -bestFace.sep));
    return out;
}

// Sphere vs sphere: the simplest collision test there is.
function sphereSphere(A, B) {
    const d = sub(B.pos, A.pos);
    const rsum = A.shape.r + B.shape.r;
    const d2 = len2(d);
    if (d2 >= rsum * rsum) return [];
    const dist = Math.sqrt(d2) || 1e-9;
    const n = mul(d, 1 / dist);
    const p = add(A.pos, mul(n, A.shape.r - (rsum - dist) / 2));   // mid-overlap
    return [{ a: A, b: B, p, n, pen: rsum - dist, e: E_CUBE, mu: MU_CUBE }];
}

// Sphere vs convex: find the closest point on the hull to the sphere's
// center. Checking the faces the center lies in front of (projections onto
// the face when inside its polygon, else its boundary edges) covers face,
// edge, and corner contact uniformly.
function sphereConvex(S, C) {
    const verts = worldVerts(C);
    const P = S.pos, r = S.shape.r;

    let inside = true, deepSep = -Infinity, deepN = null;
    const front = [];
    for (const f of C.shape.faces) {
        const n = rotateLocal(C, f.n);
        const sep = dot(n, P) - dot(n, verts[f.v[0]]);
        if (sep >= r) return [];   // a face plane separates by more than r
        if (sep > 0) { inside = false; front.push({ f, n, sep }); }
        else if (sep > deepSep) { deepSep = sep; deepN = n; }
    }
    if (inside) {   // center swallowed by the hull: push out through nearest face
        const q = sub(P, mul(deepN, deepSep));
        return [{ a: S, b: C, p: q, n: neg(deepN), pen: r - deepSep, e: E_CUBE, mu: MU_CUBE }];
    }

    let bestQ = null, bestD2 = Infinity;
    for (const { f, n, sep } of front) {
        const q = sub(P, mul(n, sep));   // projection onto the face plane
        let inPoly = true;
        for (let k = 0; k < f.v.length; k++) {
            const p1 = verts[f.v[k]], p2 = verts[f.v[(k + 1) % f.v.length]];
            if (dot(sub(q, p1), cross(sub(p2, p1), n)) > 0) { inPoly = false; break; }
        }
        if (inPoly) {
            if (sep * sep < bestD2) { bestD2 = sep * sep; bestQ = q; }
            continue;
        }
        for (let k = 0; k < f.v.length; k++) {   // nearest point on the face's edges
            const p1 = verts[f.v[k]], p2 = verts[f.v[(k + 1) % f.v.length]];
            const ev = sub(p2, p1);
            const t = clampN(dot(sub(P, p1), ev) / len2(ev), 0, 1);
            const cand = add(p1, mul(ev, t));
            const d2c = len2(sub(P, cand));
            if (d2c < bestD2) { bestD2 = d2c; bestQ = cand; }
        }
    }
    if (!bestQ || bestD2 >= r * r) return [];
    const dist = Math.sqrt(bestD2) || 1e-9;
    return [{ a: S, b: C, p: bestQ, n: mul(sub(bestQ, P), 1 / dist), pen: r - dist, e: E_CUBE, mu: MU_CUBE }];
}

// Narrowphase dispatch by shape pair (contacts always use n pointing A -> B).
function collide(A, B) {
    const aSphere = A.shape.kind === 'sphere', bSphere = B.shape.kind === 'sphere';
    if (aSphere && bSphere) return sphereSphere(A, B);
    if (aSphere) return sphereConvex(A, B);
    if (bSphere) return sphereConvex(B, A).map(c => ({ ...c, a: A, b: B, n: neg(c.n) }));
    return convexConvex(A, B);
}

// ================ contact solver (sequential impulses) ==============
// Contact normal points from body `a` to body `b`; a === null means the
// static floor. Positive normal impulse pushes b along +n and a along -n.
// Sleeping bodies get zero inverse mass so they act as static geometry.

function relVel(c) {
    const vb = add(c.b.vel, cross(c.b.angVel, c.rb));
    return c.a ? sub(vb, add(c.a.vel, cross(c.a.angVel, c.ra))) : vb;
}
function applyImpulse(c, P) {
    if (c.imB) {
        c.b.vel = add(c.b.vel, mul(P, c.imB));
        c.b.angVel = add(c.b.angVel, mul(cross(c.rb, P), c.iiB));
    }
    if (c.imA) {
        c.a.vel = sub(c.a.vel, mul(P, c.imA));
        c.a.angVel = sub(c.a.angVel, mul(cross(c.ra, P), c.iiA));
    }
}
function effMass(c, dir) {
    let k = c.imB + c.iiB * len2(cross(c.rb, dir));
    if (c.a) k += c.imA + c.iiA * len2(cross(c.ra, dir));
    return 1 / k;
}

function prepContact(c, h) {
    c.imB = c.b.sleeping ? 0 : c.b.invMass;
    c.iiB = c.b.sleeping ? 0 : c.b.invI;
    c.imA = (c.a && !c.a.sleeping) ? c.a.invMass : 0;
    c.iiA = (c.a && !c.a.sleeping) ? c.a.invI : 0;
    c.rb = sub(c.p, c.b.pos);
    c.ra = c.a ? sub(c.p, c.a.pos) : null;
    c.t1 = norm(Math.abs(c.n.x) > 0.9 ? cross(c.n, v3(0, 1, 0)) : cross(c.n, v3(1, 0, 0)));
    c.t2 = cross(c.n, c.t1);
    c.massN  = effMass(c, c.n);
    c.massT1 = effMass(c, c.t1);
    c.massT2 = effMass(c, c.t2);
    const vn0 = dot(relVel(c), c.n);
    const bias = Math.min(MAX_BIAS, (BAUMGARTE / h) * Math.max(0, c.pen - PEN_SLOP));
    const rest = vn0 < -REST_THRESHOLD ? -c.e * vn0 : 0;
    c.target = Math.max(bias, rest);
    c.accN = 0; c.accT1 = 0; c.accT2 = 0;
}

// Add dLambda to the accumulated impulse c[acc], clamp the running total,
// and apply only the difference — the standard accumulated-clamping trick
// that lets an iteration back off without ever violating the bounds.
function clampedImpulse(c, dir, dLambda, acc, lo, hi) {
    const old = c[acc];
    c[acc] = clampN(old + dLambda, lo, hi);
    const applied = c[acc] - old;
    if (applied !== 0) applyImpulse(c, mul(dir, applied));
}

function solveContact(c) {
    // normal: push the approach velocity up to the target, never pulling
    clampedImpulse(c, c.n, (c.target - dot(relVel(c), c.n)) * c.massN, 'accN', 0, Infinity);
    // friction: drive tangential velocity to zero within the Coulomb cone
    const maxF = c.mu * c.accN;
    clampedImpulse(c, c.t1, -dot(relVel(c), c.t1) * c.massT1, 'accT1', -maxF, maxF);
    clampedImpulse(c, c.t2, -dot(relVel(c), c.t2) * c.massT2, 'accT2', -maxF, maxF);
}

// ============================ simulation ===========================
function physStep(h) {
    let anyAwake = false;
    for (const b of bodies) {
        b.touch = false;
        if (b.sleeping) continue;
        anyAwake = true;
        b.vel.y += GRAVITY * h;
        b.vel = mul(b.vel, 1 - LIN_DAMP * h);
        b.angVel = mul(b.angVel, 1 - ANG_DAMP * h);
    }
    if (!anyAwake) return;   // a fully sleeping scene costs nothing

    const grid = buildGrid();
    const contacts = [];

    // floor contacts: only for vertices actually over the finite plane,
    // so cubes tip over and fall off the edge naturally. Two edge guards:
    // 1) a body with any vertex driven deep below the plane while over it
    //    has fallen past the edge and is UNDER the floor — a thin plane
    //    never pushes from below, so it gets no floor contacts at all
    //    (deep overhanging vertices don't count, keeping edge pivots);
    // 2) a submerged vertex closer to the plane's side boundary than to
    //    its surface (inset < depth) is past the edge, so a corner that
    //    dipped while overhanging and swung back over the floor during a
    //    roll can't fire a deep contact that kicks the cube into the air.
    for (const b of bodies) {
        if (b.sleeping || b.pos.y > b.radius + 0.1) continue;

        // Sphere vs floor: nearest point on the floor rectangle to the
        // center — one test that covers the face, the rim, and the corners
        // (rim contacts get a slanted normal, so balls roll off smoothly).
        // Contact only while the center is above the plane: a thin floor
        // never pushes from below.
        if (b.shape.kind === 'sphere') {
            if (b.pos.y <= 0) continue;
            const q = v3(clampN(b.pos.x, -FLOOR_HALF, FLOOR_HALF), 0,
                         clampN(b.pos.z, -FLOOR_HALF, FLOOR_HALF));
            const dq = sub(b.pos, q);
            const d2 = len2(dq);
            const r = b.shape.r;
            if (d2 >= r * r) continue;
            const dist = Math.sqrt(d2) || 1e-9;
            contacts.push({ a: null, b, p: q, n: mul(dq, 1 / dist), pen: r - dist, e: E_FLOOR, mu: MU_FLOOR });
            b.touch = true;
            continue;
        }

        const verts = worldVerts(b);
        let under = false;
        for (const v of verts) {
            if (-v.y > 0.3 && Math.abs(v.x) <= FLOOR_HALF && Math.abs(v.z) <= FLOOR_HALF) {
                under = true;
                break;
            }
        }
        if (under) continue;
        for (const v of verts) {
            if (v.y >= 0.02) continue;
            const depth = -v.y;
            const inset = Math.min(FLOOR_HALF - Math.abs(v.x), FLOOR_HALF - Math.abs(v.z));
            if (inset < 0 || inset < depth) continue;
            contacts.push({ a: null, b, p: v, n: UP, pen: Math.max(0, depth), e: E_FLOOR, mu: MU_FLOOR });
            b.touch = true;
        }
        // rim contacts: where a body edge crosses the floor boundary at or
        // below the surface, the floor's rim supports it. A straddling
        // body then pivots about the actual floor edge instead of its own
        // inner vertices, so its face can't visibly sink near the rim.
        // Penetration is capped so a sudden deep crossing during a fast
        // roll gives gentle support, never a launching correction.
        if (Math.abs(b.pos.x) + b.radius > FLOOR_HALF || Math.abs(b.pos.z) + b.radius > FLOOR_HALF) {
            for (const e of b.shape.edges) {
                const p1 = verts[e.i], p2 = verts[e.j];
                for (let axis = 0; axis < 2; axis++) {
                    const c1 = axis ? p1.z : p1.x, c2 = axis ? p2.z : p2.x;
                    for (const bound of [-FLOOR_HALF, FLOOR_HALF]) {
                        if ((c1 < bound) === (c2 < bound)) continue;
                        const t = (bound - c1) / (c2 - c1);
                        const y = p1.y + t * (p2.y - p1.y);
                        if (y >= 0.02 || -y > 0.3) continue;
                        const other = axis ? p1.x + t * (p2.x - p1.x) : p1.z + t * (p2.z - p1.z);
                        if (Math.abs(other) > FLOOR_HALF) continue;
                        const p = axis ? v3(other, y, bound) : v3(bound, y, other);
                        contacts.push({ a: null, b, p, n: UP, pen: Math.min(Math.max(0, -y), 0.05), e: E_FLOOR, mu: MU_FLOOR });
                        b.touch = true;
                    }
                }
            }
        }
    }

    // cube-cube contacts via the spatial hash; only awake bodies scan,
    // so a settled pile is never even looked at
    for (let i = 0; i < bodies.length; i++) {
        const A = bodies[i];
        if (A.sleeping) continue;   // pairs with sleepers are found from the awake side
        forEachNeighbor(grid, A, (j) => {
            const B = bodies[j];
            if (B === A || (!B.sleeping && j <= i)) return;   // visit each pair once
            const r = A.radius + B.radius;
            if (len2(sub(B.pos, A.pos)) > r * r) return;
            // a fast body crashing into a sleeper wakes it (and what it rests on)
            if (B.sleeping && len2(A.vel) > WAKE_SPEED2) wakeChain(B, grid);
            const cs = collide(A, B);
            if (cs.length) {
                A.touch = B.touch = true;
                for (const c of cs) contacts.push(c);
            }
        });
    }

    for (const c of contacts) prepContact(c, h);
    for (let it = 0; it < SOLVER_ITERS; it++) {
        for (const c of contacts) solveContact(c);
    }

    // a solved impulse that exceeds quiet weight transfer is a shove: wake
    for (const c of contacts) {
        if (c.accN < WAKE_IMPULSE) continue;
        if (c.a && c.a.sleeping) wakeChain(c.a, grid);
        if (c.b.sleeping) wakeChain(c.b, grid);
    }

    for (const b of bodies) {
        if (b.sleeping) continue;
        b.pos = add(b.pos, mul(b.vel, h));
        qIntegrate(b.q, b.angVel, h);
        b.axes = quatAxes(b.q);
        // rolling resistance: a rolling ball has zero contact velocity, so
        // friction goes silent and it would roll forever — bleed spin while
        // a sphere is touching something so it can slow down and sleep
        if (b.touch && b.shape.kind === 'sphere') {
            b.angVel = mul(b.angVel, Math.max(0, 1 - ROLL_RESIST * h));
        }
        // fall asleep only while touching something and consistently slow;
        // near-still bodies get extra damping so pile jitter dies out, and
        // a momentary spike delays sleep rather than restarting the count
        if (b.touch && len2(b.vel) < SLEEP_LIN2 && len2(b.angVel) < SLEEP_ANG2) {
            b.vel = mul(b.vel, 0.94);
            b.angVel = mul(b.angVel, 0.94);
            if (++b.sleepTimer >= SLEEP_STEPS) {
                b.sleeping = true;
                b.vel = v3(0, 0, 0);
                b.angVel = v3(0, 0, 0);
            }
        } else {
            b.sleepTimer = Math.max(0, b.sleepTimer - 4);
        }
    }

    for (let i = bodies.length - 1; i >= 0; i--) {
        if (bodies[i].pos.y < -40) {
            if (renderer) renderer.bodyRemoved(bodies[i]);
            bodies.splice(i, 1);
        }
    }
}

// ==================== physics engine interface =====================
// Two interchangeable engines simulate the same `bodies` array. Both
// implement:
//   name          label shown on the physics button
//   init()        set up; may be async; false/throw = unsupported
//   activate()    take over the scene from the other engine mid-flight
//   step(h)       advance the world; keep every body object's state current
//                 and fire renderer cache hooks for wakes/removals
//   bodyAdded(b)  a new body was pushed onto `bodies`
//   bodyDropped(b) the body cap evicted this body: wake its neighbors
//   bounce()      floor jolt
//   clear()       scene emptied
// This engine is the JavaScript one — physStep above IS its step function.
// The other lives in physics.cpp, wrapped by physics-wasm.js.
const JSPhysics = {
    name: 'JavaScript',
    init() { return true; },
    activate() {},   // body objects already carry their full dynamic state
    step: physStep,
    bodyAdded() {},
    bodyDropped(gone) {
        for (const nb of bodies) {   // don't leave sleepers floating on a removed support
            if (nb.sleeping && len2(sub(nb.pos, gone.pos)) < (nb.radius + gone.radius) ** 2) wakeBody(nb);
        }
    },
    // The floor jolts: every body pops upward along the plane normal, with a
    // little variation so a pile scatters instead of rising in step.
    bounce() {
        for (const b of bodies) {
            wakeBody(b);
            b.vel = add(b.vel, v3(0, 5.5 + Math.random() * 2.5, 0));
            b.angVel = add(b.angVel, v3((Math.random() * 2 - 1) * 1.5,
                                        (Math.random() * 2 - 1) * 1.5,
                                        (Math.random() * 2 - 1) * 1.5));
        }
    },
    clear() {}
};

// ============================= camera ==============================
const view = document.getElementById('view');
const countEl = document.getElementById('count');

const FOV = 55 * Math.PI / 180;
const MOVE_SPEED = 10;         // WASD / arrows, units/s
const ZOOM_SPEED = 10;         // +/- keys, units/s
const LOOK_SENS  = 0.005;      // radians per pixel of mouse drag
const SPACE_INTERVAL = 0.15;   // held-space drop rate: one cube per 0.15s

let camPos = v3(0, 7.5, 17);
let yaw = 0, pitch = -0.34;    // matches the original fixed view of the floor
let camF, camR, camU, projScale, W, H;
let canvas = null;             // the active renderer's canvas

// Camera basis from yaw/pitch (fly camera). Pitch is clamped well short of
// the poles so cross(camF, UP) never degenerates.
function updateView() {
    const cp = Math.cos(pitch);
    camF = v3(cp * Math.sin(yaw), Math.sin(pitch), -cp * Math.cos(yaw));
    camR = norm(cross(camF, UP));
    camU = cross(camR, camF);
}

function resizeView() {
    W = window.innerWidth;
    H = window.innerHeight;
    projScale = (H / 2) / Math.tan(FOV / 2);
    if (canvas) {
        canvas.width = W;
        canvas.height = H;
    }
    if (renderer) renderer.resize();
}
window.addEventListener('resize', resizeView);

// Perspective-project a world point; null when behind the near limit.
// (Used by the mouse ray and the software renderer; the GPU renderers do
// the same job with a view-projection matrix in their shaders.)
function project(p) {
    const rel = sub(p, camPos);
    const z = dot(rel, camF);
    if (z < 0.1) return null;
    return {
        x: W / 2 + dot(rel, camR) / z * projScale,
        y: H / 2 - dot(rel, camU) / z * projScale,
        iz: 1 / z   // 1/depth is linear in screen space: exact for flat faces
    };
}

// ================= shared geometry for the GPU renderers ===========

// Unit cube mesh: 24 vertices (4 per face so each face carries its own
// normal for flat shading) + 36 indices, wound counter-clockwise seen from
// outside. Interleaved [x,y,z, nx,ny,nz] per vertex.
function buildCubeMesh() {
    const verts = [], idx = [];
    const faces = [   // [normal, u, v] with u x v = normal
        [v3(1, 0, 0),  v3(0, 1, 0), v3(0, 0, 1)],
        [v3(-1, 0, 0), v3(0, 0, 1), v3(0, 1, 0)],
        [v3(0, 1, 0),  v3(0, 0, 1), v3(1, 0, 0)],
        [v3(0, -1, 0), v3(1, 0, 0), v3(0, 0, 1)],
        [v3(0, 0, 1),  v3(1, 0, 0), v3(0, 1, 0)],
        [v3(0, 0, -1), v3(0, 1, 0), v3(1, 0, 0)]
    ];
    for (const [n, u, w] of faces) {
        const base = verts.length / 6;
        for (const [su, sw] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
            verts.push(
                (n.x + su * u.x + sw * w.x) * CUBE_HALF,
                (n.y + su * u.y + sw * w.y) * CUBE_HALF,
                (n.z + su * u.z + sw * w.z) * CUBE_HALF,
                n.x, n.y, n.z);
        }
        idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    return { verts: new Float32Array(verts), indices: new Uint16Array(idx) };
}

// Regular tetrahedron mesh: 4 triangular faces with flat normals, wound CCW
// from outside (same local vertices the physics descriptor uses).
function buildTetMesh() {
    const verts = [], idx = [];
    for (const f of SHAPES.tet.faces) {   // descriptor windings are already outward
        const base = verts.length / 6;
        for (const i of f.v) {
            const v = tetVerts[i];
            verts.push(v.x, v.y, v.z, f.n.x, f.n.y, f.n.z);
        }
        idx.push(base, base + 1, base + 2);
    }
    return { verts: new Float32Array(verts), indices: new Uint16Array(idx) };
}

// UV sphere mesh: positions on the sphere double as smooth normals.
function buildSphereMesh() {
    const stacks = 12, slices = 18;
    const verts = [], idx = [];
    for (let i = 0; i <= stacks; i++) {
        const phi = (i / stacks) * Math.PI;         // 0 at +y pole
        for (let j = 0; j <= slices; j++) {
            const th = (j / slices) * 2 * Math.PI;
            const n = v3(Math.sin(phi) * Math.cos(th), Math.cos(phi), Math.sin(phi) * Math.sin(th));
            verts.push(n.x * SPHERE_R, n.y * SPHERE_R, n.z * SPHERE_R, n.x, n.y, n.z);
        }
    }
    for (let i = 0; i < stacks; i++) {
        for (let j = 0; j < slices; j++) {
            const a = i * (slices + 1) + j, b = a + slices + 1;
            idx.push(a, b, a + 1, a + 1, b, b + 1);
        }
    }
    return { verts: new Float32Array(verts), indices: new Uint16Array(idx) };
}

// Floor grid line segments, lifted slightly to avoid z-fighting the floor.
function buildGridLines() {
    const y = 0.003, pts = [];
    for (let g = -FLOOR_HALF; g <= FLOOR_HALF; g += 2) {
        pts.push(g, y, -FLOOR_HALF, g, y, FLOOR_HALF);
        pts.push(-FLOOR_HALF, y, g, FLOOR_HALF, y, g);
    }
    return new Float32Array(pts);
}

// ============================== input ==============================
// Click = drop a cube; hold Space to rain them at a paced rate (both use
// the pointer position). Hold Shift and move the mouse to look around —
// clicks are ignored while Shift is held so a look never drops a cube.
// WASD flies, up/down arrows pedestal along the floor normal, left/right
// arrows strafe, wheel / pinch / + - = _ zoom, B bounces the floor's
// cubes upward, C clears the scene, R cycles the renderer.

// Unproject the click into a world ray and drop a cube where it hits the floor.
function dropCubeAt(clientX, clientY) {
    const sx = (clientX - W / 2) / projScale;   // view-space ray slopes
    const sy = (H / 2 - clientY) / projScale;
    const dir = norm(add(camF, add(mul(camR, sx), mul(camU, sy))));
    if (dir.y >= -1e-4) return;                  // ray never reaches the floor
    const t = -camPos.y / dir.y;
    const hit = add(camPos, mul(dir, t));
    spawnBody(clampN(hit.x, -FLOOR_HALF + 1, FLOOR_HALF - 1),
              clampN(hit.z, -FLOOR_HALF + 1, FLOOR_HALF - 1));
}

const SHAPE_ORDER = ['cube', 'sphere', 'tet'];
function nextShape() {
    spawnShape = SHAPE_ORDER[(SHAPE_ORDER.indexOf(spawnShape) + 1) % SHAPE_ORDER.length];
    shapeBtn.textContent = 'shape: ' + spawnShape;
}

const keys = Object.create(null);
let lastMouseX = -1, lastMouseY = -1;   // -1 until the pointer first moves
let pendingDX = 0, pendingDY = 0, pendingOrbit = 0, wheelDolly = 0;
let spaceTimer = 0;                     // paces held-space cube drops

function dropAtPointer() {
    dropCubeAt(lastMouseX >= 0 ? lastMouseX : W / 2,
               lastMouseY >= 0 ? lastMouseY : H / 2);
}

function clearScene() {
    bodies.length = 0;
    physics.clear();
    if (renderer) renderer.sceneCleared();
}

function bounceAll() {
    physics.bounce();
}

// on-screen buttons (phones have no B/C/R keys); blur after use so a later
// Space press drops a cube instead of re-triggering the focused button
document.getElementById('bounceBtn').addEventListener('click', (e) => {
    bounceAll();
    e.currentTarget.blur();
});
document.getElementById('clearBtn').addEventListener('click', (e) => {
    clearScene();
    e.currentTarget.blur();
});
const gfxBtn = document.getElementById('gfxBtn');
gfxBtn.addEventListener('click', (e) => {
    nextRenderer();
    e.currentTarget.blur();
});
const shapeBtn = document.getElementById('shapeBtn');
shapeBtn.addEventListener('click', (e) => {
    nextShape();
    e.currentTarget.blur();
});
const physBtn = document.getElementById('physBtn');
physBtn.addEventListener('click', (e) => {
    nextPhysics();
    e.currentTarget.blur();
});

// Fullscreen toggle, with WebKit-prefixed fallbacks for older mobile
// Safari. On browsers without the API (iPhone Safari) it fails silently.
// Entering/leaving fullscreen fires a resize, which already rebuilds the
// canvas and every renderer's buffers.
const fullBtn = document.getElementById('fullBtn');
const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;
fullBtn.addEventListener('click', (e) => {
    e.currentTarget.blur();
    try {
        if (fsElement()) {
            (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        } else {
            const root = document.documentElement;
            const req = root.requestFullscreen || root.webkitRequestFullscreen;
            if (req) {
                const p = req.call(root);
                if (p && p.catch) p.catch(() => {});
            }
        }
    } catch (err) {}
});
const fsLabel = () => { fullBtn.textContent = fsElement() ? 'exit full' : 'fullscreen'; };
document.addEventListener('fullscreenchange', fsLabel);
document.addEventListener('webkitfullscreenchange', fsLabel);

window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    keys[k] = true;
    if (k === ' ') {
        e.preventDefault();
        if (!e.repeat) {   // first press drops now; frame() paces the rest
            spaceTimer = 0;
            dropAtPointer();
        }
    } else if (k.startsWith('arrow')) {
        e.preventDefault();
    } else if (k === 'c' && !e.repeat) {
        clearScene();
    } else if (k === 'b' && !e.repeat) {
        bounceAll();
    } else if (k === 'r' && !e.repeat) {
        nextRenderer();
    } else if (k === 't' && !e.repeat) {
        nextShape();
    } else if (k === 'p' && !e.repeat) {
        nextPhysics();
    }
});
window.addEventListener('keyup', (e) => { delete keys[e.key.toLowerCase()]; });
window.addEventListener('blur', () => { for (const k in keys) delete keys[k]; });   // no stuck keys

window.addEventListener('mousemove', (e) => {
    if (keys.shift && lastMouseX >= 0) {          // hold Shift to look around
        pendingDX += e.clientX - lastMouseX;
        pendingDY += e.clientY - lastMouseY;
    } else if (keys.control && lastMouseX >= 0) { // hold Ctrl to orbit the scene
        pendingOrbit += e.clientX - lastMouseX;
    }
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
});
view.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || keys.shift || keys.control) return;   // no drops mid-look/orbit
    dropCubeAt(e.clientX, e.clientY);
});
// mouse wheel and trackpad two-finger scroll; trackpad pinch arrives as
// ctrl+wheel with small deltas, so it gets a stronger factor
view.addEventListener('wheel', (e) => {
    e.preventDefault();
    wheelDolly -= e.deltaY * (e.ctrlKey ? 0.03 : 0.012);
}, { passive: false });

// Consume the accumulated input once per frame; returns true if the
// camera changed (which tells the software renderer its cache is stale).
function applyCameraInput(dt) {
    let changed = false;
    if (pendingDX || pendingDY) {
        // reversed ("grab the world"): mouse right turns the view left
        yaw -= pendingDX * LOOK_SENS;
        pitch = clampN(pitch + pendingDY * LOOK_SENS, -1.45, 1.45);
        pendingDX = 0;
        pendingDY = 0;
        updateView();
        changed = true;
    }
    if (pendingOrbit) {
        // Orbit: rotate the camera rig — position AND facing together — about
        // the floor's central vertical axis. Rotating the rig one way makes
        // the whole scene appear to spin the other, so this is equivalent to
        // "rotating the scene" while the physics world stays put. The sign
        // matches the grab-the-world look: mouse right spins the scene right.
        const a = pendingOrbit * LOOK_SENS;
        pendingOrbit = 0;
        const c = Math.cos(a), s = Math.sin(a);
        camPos = v3(c * camPos.x + s * camPos.z, camPos.y, -s * camPos.x + c * camPos.z);
        yaw -= a;
        updateView();
        changed = true;
    }
    let fwd = 0, side = 0, elev = 0, dolly = wheelDolly;
    wheelDolly = 0;
    if (keys.w) fwd += 1;
    if (keys.s) fwd -= 1;
    if (keys.d) side += 1;
    if (keys.a) side -= 1;
    if (keys.arrowright) side += 1;       // strafe (camR is always horizontal)
    if (keys.arrowleft) side -= 1;
    if (keys.arrowup) elev += 1;          // pedestal: straight up the floor normal
    if (keys.arrowdown) elev -= 1;
    if (keys['+'] || keys['=']) dolly += ZOOM_SPEED * dt;
    if (keys['-'] || keys['_']) dolly -= ZOOM_SPEED * dt;
    if (fwd || side || elev || dolly) {
        const mv = MOVE_SPEED * dt;
        camPos = add(camPos, add(mul(camF, fwd * mv + dolly),
                                 add(mul(camR, side * mv), v3(0, elev * mv, 0))));
        camPos.x = clampN(camPos.x, -80, 80);
        camPos.y = clampN(camPos.y, 0.7, 60);   // never below the floor
        camPos.z = clampN(camPos.z, -80, 80);
        changed = true;
    }
    return changed;
}

// ========================= renderer registry ========================
// A canvas is permanently bound to its first context type, so switching
// renderers swaps in a fresh canvas element. Unsupported renderers (no
// WebGL2 / no WebGPU) are skipped automatically when cycling.
const RENDERERS = [SoftwareRenderer, WebGLRenderer, WebGPURenderer];
let renderer = null, rendererIndex = -1, switching = false;

async function useRenderer(index) {
    if (switching) return;
    switching = true;
    for (let tries = 0; tries < RENDERERS.length; tries++) {
        const pick = (index + tries) % RENDERERS.length;
        const target = RENDERERS[pick];
        const cv = document.createElement('canvas');
        cv.width = W;
        cv.height = H;
        let ok = false;
        try {
            ok = await target.init(cv);
        } catch (err) {
            console.warn(target.name + ' renderer unavailable:', err.message);
        }
        if (!ok) continue;
        if (renderer) renderer.dispose();
        view.replaceChildren(cv);
        canvas = cv;
        renderer = target;
        rendererIndex = pick;
        gfxBtn.textContent = 'renderer: ' + target.name;
        break;
    }
    switching = false;
}

function nextRenderer() {
    useRenderer((rendererIndex + 1) % RENDERERS.length);
}

// ===================== physics engine registry ======================
// Same pattern as the renderers: cycle engines live, skipping any that
// fail to initialize (e.g. the WASM module missing or unsupported). The
// scene transfers mid-flight: body objects persist, and activate() hands
// their current state to the incoming engine.
const PHYSICS_ENGINES = [JSPhysics, WasmPhysics];
let physics = JSPhysics;
let physicsIndex = 0, physicsSwitching = false;

async function usePhysics(index) {
    if (physicsSwitching) return;
    physicsSwitching = true;
    for (let tries = 0; tries < PHYSICS_ENGINES.length; tries++) {
        const pick = (index + tries) % PHYSICS_ENGINES.length;
        const target = PHYSICS_ENGINES[pick];
        let ok = false;
        try {
            ok = await target.init();
        } catch (err) {
            console.warn(target.name + ' physics unavailable:', err.message);
        }
        if (!ok) continue;
        target.activate();   // adopt the scene from the previous engine
        physics = target;
        physicsIndex = pick;
        physBtn.textContent = 'physics: ' + target.name;
        break;
    }
    physicsSwitching = false;
}

function nextPhysics() {
    usePhysics((physicsIndex + 1) % PHYSICS_ENGINES.length);
}

// ============================ main loop ============================
let last = performance.now(), acc = 0;
let fpsFrames = 0, fpsTime = 0, fpsValue = 0;   // fps over half-second windows
function frame(now) {
    const rawDt = (now - last) / 1000;          // unclamped: honest about slow frames
    const dtReal = Math.min(0.05, rawDt);
    last = now;
    acc += dtReal;

    fpsFrames++;
    fpsTime += rawDt;
    if (fpsTime >= 0.5) {
        fpsValue = Math.round(fpsFrames / fpsTime);
        fpsFrames = 0;
        fpsTime = 0;
    }

    if (keys[' ']) {   // held space rains cubes at a controlled pace
        spaceTimer += dtReal;
        if (spaceTimer >= SPACE_INTERVAL) {
            spaceTimer -= SPACE_INTERVAL;
            dropAtPointer();
        }
    }

    const camMoved = applyCameraInput(dtReal);

    let steps = 0;
    while (acc >= FIXED_DT && steps < 6) { physics.step(FIXED_DT); acc -= FIXED_DT; steps++; }
    if (steps === 6) acc = 0;    // dropped frames: don't spiral

    if (renderer && !switching) renderer.render(camMoved);

    let awake = 0;
    for (const b of bodies) if (!b.sleeping) awake++;
    countEl.textContent = 'bodies: ' + bodies.length + '  (' + awake + ' awake)  ·  ' + fpsValue + ' fps';
    requestAnimationFrame(frame);
}

// ============================== boot ===============================
updateView();
resizeView();
useRenderer(0);
spawnBody(0, 0);
requestAnimationFrame(frame);
