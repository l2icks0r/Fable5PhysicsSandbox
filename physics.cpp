// Fable 5 Physics Sandbox — created by Claude Fable 5 (Anthropic), directed by
// the repository author. MIT License; see LICENSE and README.md.
// ===========================================================================
// PHYSICS ENGINE — C++ port of the JavaScript engine in simulation.js,
// compiled to WebAssembly (see build-wasm.bat).
//
// This is a deliberate line-for-line translation: every constant, rule, and
// algorithm matches simulation.js so the two engines are interchangeable at
// runtime. Read them side by side — the interesting differences are not the
// physics but the *memory model*: JavaScript builds millions of short-lived
// {x,y,z} objects that the garbage collector must chase, while here every
// body, contact, and grid cell lives in flat static arrays fixed at compile
// time. No allocation ever happens after startup. That — not the arithmetic
// — is where most of the speedup comes from.
//
// The JavaScript side (physics-wasm.js) drives this module through a handful
// of exported functions and reads results back through one packed state
// buffer, viewed directly as a typed array over the WASM linear memory.
// ===========================================================================

#include <emscripten.h>

typedef double f64;
typedef unsigned int u32;

#define EXPORT extern "C" EMSCRIPTEN_KEEPALIVE

static inline f64 fsqrt(f64 x) { return __builtin_sqrt(x); }
static inline f64 fabsd(f64 x) { return __builtin_fabs(x); }
static inline f64 ffloor(f64 x) { return __builtin_floor(x); }
static inline f64 fmind(f64 a, f64 b) { return a < b ? a : b; }
static inline f64 fmaxd(f64 a, f64 b) { return a > b ? a : b; }
static inline f64 clampd(f64 x, f64 lo, f64 hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ============================== vec3 ==============================
struct V3 { f64 x, y, z; };
static inline V3 v3(f64 x, f64 y, f64 z) { return { x, y, z }; }
static inline V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(V3 a, f64 s) { return { a.x * s, a.y * s, a.z * s }; }
static inline V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static inline f64 dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static inline f64 len2(V3 a) { return dot(a, a); }
static inline V3 norm(V3 a) {
    f64 l = fsqrt(len2(a));
    if (l == 0) l = 1;
    return mul(a, 1.0 / l);
}

// ============================ quaternion ===========================
struct Quat { f64 w, x, y, z; };

static inline Quat qMul(Quat a, Quat b) {
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}
static inline void qNormalize(Quat& q) {
    f64 l = fsqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (l == 0) l = 1;
    q.w /= l; q.x /= l; q.y /= l; q.z /= l;
}

// ============================ constants ============================
// MUST MATCH simulation.js — the two engines are interchangeable.
static const f64 GRAVITY        = -9.81;
static const f64 CUBE_HALF      = 0.5;
static const f64 CUBE_MASS      = 1.0;
static const f64 FLOOR_HALF     = 12.0;
static const int MAX_BODIES     = 1024;

static const f64 E_FLOOR   = 0.22;
static const f64 E_CUBE    = 0.12;
static const f64 MU_FLOOR  = 0.45;
static const f64 MU_CUBE   = 0.35;

static const int SOLVER_ITERS   = 10;
static const f64 BAUMGARTE      = 0.2;
static const f64 MAX_BIAS       = 4.0;
static const f64 PEN_SLOP       = 0.01;
static const f64 REST_THRESHOLD = 0.8;
static const f64 LIN_DAMP       = 0.01;
static const f64 ANG_DAMP       = 0.05;
static const f64 ROLL_RESIST    = 1.2;

static const f64 SLEEP_LIN2   = 0.02;
static const f64 SLEEP_ANG2   = 0.04;
static const int SLEEP_STEPS  = 60;
static const f64 WAKE_SPEED2  = 1.0;
static const f64 WAKE_IMPULSE = 1.0;

static const f64 TET_S    = 0.55;
static const f64 SPHERE_R = 0.5;

// shape kinds shared with the JavaScript wrapper
enum { KIND_CUBE = 0, KIND_SPHERE = 1, KIND_TET = 2 };

// ======================== shape descriptors ========================
struct Face { int n; int v[4]; V3 normal; };
struct Edge { int i, j; };
struct Shape {
    bool isSphere;
    int vertCount;  V3 verts[8];
    int faceCount;  Face faces[6];
    int edgeCount;  Edge edges[12];
    int dirCount;   V3 edgeDirs[6];
    f64 radius, invI, r;   // r = sphere radius
};
static Shape shapes[3];

// Port of makeConvexShape: outward normals (winding auto-corrected),
// unique edges, deduplicated edge directions, bounding radius.
static void buildConvex(Shape& s, f64 invI) {
    s.isSphere = false;
    s.invI = invI;
    s.radius = 0;
    for (int i = 0; i < s.vertCount; i++) s.radius = fmaxd(s.radius, fsqrt(len2(s.verts[i])));
    for (int f = 0; f < s.faceCount; f++) {
        Face& fc = s.faces[f];
        V3 n = norm(cross(sub(s.verts[fc.v[1]], s.verts[fc.v[0]]),
                          sub(s.verts[fc.v[2]], s.verts[fc.v[0]])));
        if (dot(n, s.verts[fc.v[0]]) < 0) {   // origin-centered: flip inward windings
            for (int k = 0; k < fc.n / 2; k++) {
                int t = fc.v[k]; fc.v[k] = fc.v[fc.n - 1 - k]; fc.v[fc.n - 1 - k] = t;
            }
            n = neg(n);
        }
        fc.normal = n;
    }
    s.edgeCount = 0;
    s.dirCount = 0;
    for (int f = 0; f < s.faceCount; f++) {
        const Face& fc = s.faces[f];
        for (int k = 0; k < fc.n; k++) {
            int i = fc.v[k], j = fc.v[(k + 1) % fc.n];
            int lo = i < j ? i : j, hi = i < j ? j : i;
            bool seen = false;
            for (int e = 0; e < s.edgeCount; e++) {
                if (s.edges[e].i == lo && s.edges[e].j == hi) { seen = true; break; }
            }
            if (seen) continue;
            s.edges[s.edgeCount].i = lo;
            s.edges[s.edgeCount].j = hi;
            s.edgeCount++;
            V3 dir = norm(sub(s.verts[hi], s.verts[lo]));
            bool dup = false;
            for (int d = 0; d < s.dirCount; d++) {
                if (len2(cross(s.edgeDirs[d], dir)) < 1e-6) { dup = true; break; }
            }
            if (!dup) s.edgeDirs[s.dirCount++] = dir;
        }
    }
}

static void buildShapes() {
    // cube: vertex i has bit0 -> +x, bit1 -> +y, bit2 -> +z
    Shape& c = shapes[KIND_CUBE];
    c.vertCount = 8;
    for (int i = 0; i < 8; i++) {
        c.verts[i] = v3((i & 1) ? CUBE_HALF : -CUBE_HALF,
                        (i & 2) ? CUBE_HALF : -CUBE_HALF,
                        (i & 4) ? CUBE_HALF : -CUBE_HALF);
    }
    const int cubeFaces[6][4] = {
        {1, 3, 7, 5}, {0, 2, 6, 4}, {2, 3, 7, 6}, {0, 1, 5, 4}, {4, 5, 7, 6}, {0, 1, 3, 2}
    };
    c.faceCount = 6;
    for (int f = 0; f < 6; f++) {
        c.faces[f].n = 4;
        for (int k = 0; k < 4; k++) c.faces[f].v[k] = cubeFaces[f][k];
    }
    buildConvex(c, 6.0 / (CUBE_MASS * (2 * CUBE_HALF) * (2 * CUBE_HALF)));

    // regular tetrahedron (Platonic solid: isotropic inertia, like the cube)
    Shape& t = shapes[KIND_TET];
    t.vertCount = 4;
    t.verts[0] = v3(TET_S, TET_S, TET_S);
    t.verts[1] = v3(TET_S, -TET_S, -TET_S);
    t.verts[2] = v3(-TET_S, TET_S, -TET_S);
    t.verts[3] = v3(-TET_S, -TET_S, TET_S);
    const int tetFaces[4][3] = { {0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {1, 3, 2} };
    t.faceCount = 4;
    for (int f = 0; f < 4; f++) {
        t.faces[f].n = 3;
        for (int k = 0; k < 3; k++) t.faces[f].v[k] = tetFaces[f][k];
    }
    buildConvex(t, 2.5 / (CUBE_MASS * TET_S * TET_S));

    // solid sphere: I = (2/5) m r^2
    Shape& s = shapes[KIND_SPHERE];
    s.isSphere = true;
    s.r = SPHERE_R;
    s.radius = SPHERE_R;
    s.invI = 2.5 / (CUBE_MASS * SPHERE_R * SPHERE_R);
}

// ============================== bodies =============================
struct Body {
    int id, kind;
    bool sleeping, touch;
    int sleepTimer;
    V3 pos, vel, angVel;
    Quat q;
    V3 ax0, ax1, ax2;   // orientation axes (columns of the rotation matrix)
    f64 invI, radius;
    int cx, cy, cz;     // spatial-hash cell
};
static Body bodies[MAX_BODIES];
static int bodyCount = 0;

static inline void updateAxes(Body& b) {
    f64 w = b.q.w, x = b.q.x, y = b.q.y, z = b.q.z;
    b.ax0 = v3(1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w));
    b.ax1 = v3(2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w));
    b.ax2 = v3(2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y));
}
static inline V3 rotLocal(const Body& b, V3 v) {
    return add(mul(b.ax0, v.x), add(mul(b.ax1, v.y), mul(b.ax2, v.z)));
}
static int worldVertsOf(const Body& b, V3* out) {
    const Shape& s = shapes[b.kind];
    for (int i = 0; i < s.vertCount; i++) out[i] = add(b.pos, rotLocal(b, s.verts[i]));
    return s.vertCount;
}

// ================== broadphase: uniform spatial hash ================
// Same cell scheme as simulation.js, but as a fixed open-addressed hash
// table with per-body chain links — zero allocation.
static const f64 CELL = 2 * 0.8660254037844386;   // 2 * cube bounding radius
static const int GRID_SLOTS = 4096;               // power of two
static int gridKey[GRID_SLOTS];
static int gridHead[GRID_SLOTS];
static int gridNext[MAX_BODIES];

static int gridSlot(int key, bool insert) {
    u32 s = ((u32)key * 2654435761u) & (GRID_SLOTS - 1);
    while (gridKey[s] != key) {
        if (gridKey[s] == -1) {
            if (!insert) return -1;
            gridKey[s] = key;
            break;
        }
        s = (s + 1) & (GRID_SLOTS - 1);
    }
    return (int)s;
}

static void buildGrid() {
    for (int s = 0; s < GRID_SLOTS; s++) { gridKey[s] = -1; gridHead[s] = -1; }
    for (int i = 0; i < bodyCount; i++) {
        Body& b = bodies[i];
        b.cx = (int)ffloor(b.pos.x / CELL) + 512;
        b.cy = (int)ffloor(b.pos.y / CELL) + 512;
        b.cz = (int)ffloor(b.pos.z / CELL) + 512;
        int slot = gridSlot((b.cx << 20) | (b.cy << 10) | b.cz, true);
        gridNext[i] = gridHead[slot];
        gridHead[slot] = i;
    }
}

template <typename F>
static void forNeighbors(const Body& b, F visit) {
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int slot = gridSlot(((b.cx + dx) << 20) | ((b.cy + dy) << 10) | (b.cz + dz), false);
                if (slot < 0) continue;
                for (int j = gridHead[slot]; j != -1; j = gridNext[j]) visit(j);
            }
        }
    }
}

static void wakeBody(Body& b) {
    b.sleeping = false;
    b.sleepTimer = 0;
}

static void wakeChain(Body& b) {
    wakeBody(b);
    forNeighbors(b, [&](int j) {   // one level: wake whatever rests against it
        Body& nb = bodies[j];
        f64 rr = nb.radius + b.radius;
        if (nb.sleeping && len2(sub(nb.pos, b.pos)) < rr * rr) wakeBody(nb);
    });
}

// ============================ contacts =============================
struct Contact {
    int a, b;   // body indices; a == -1 means the static floor
    V3 p, n, t1, t2, ra, rb;
    f64 pen, e, mu;
    f64 imA, iiA, imB, iiB;
    f64 massN, massT1, massT2, target, accN, accT1, accT2;
};
static const int MAX_CONTACTS = 16384;
static Contact contacts[MAX_CONTACTS];
static int contactCount = 0;

static Contact* pushContact(int a, int b, V3 p, V3 n, f64 pen, f64 e, f64 mu) {
    if (contactCount >= MAX_CONTACTS) return 0;
    Contact& c = contacts[contactCount++];
    c.a = a; c.b = b; c.p = p; c.n = n; c.pen = pen; c.e = e; c.mu = mu;
    return &c;
}

// ===================== collision: convex vs convex ==================
// Same algorithm as simulation.js: SAT over face normals + edge-direction
// crosses, then reference-face clipping or supporting-edge closest points.

static int clipPolyBuf(const V3* in, int n, V3 pn, f64 d, V3* out) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        V3 p = in[i], q = in[(i + 1) % n];
        f64 dp = dot(p, pn) - d, dq = dot(q, pn) - d;
        if (dp <= 0) out[m++] = p;
        if (dp * dq < 0) out[m++] = add(p, mul(sub(q, p), dp / (dp - dq)));
    }
    return m;
}

static void closestSegSeg(V3 p1, V3 q1, V3 p2, V3 q2, V3& c1, V3& c2) {
    V3 d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
    f64 a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    f64 c = dot(d1, r), b = dot(d1, d2);
    f64 denom = a * e - b * b;
    f64 s = denom > 1e-8 ? clampd((b * f - c * e) / denom, 0, 1) : 0;
    f64 t = (b * s + f) / e;
    if (t < 0)      { t = 0; s = clampd(-c / a, 0, 1); }
    else if (t > 1) { t = 1; s = clampd((b - c) / a, 0, 1); }
    c1 = add(p1, mul(d1, s));
    c2 = add(p2, mul(d2, t));
}

static void convexConvex(int ia, int ib) {
    Body& A = bodies[ia];
    Body& B = bodies[ib];
    const Shape& sa = shapes[A.kind];
    const Shape& sb = shapes[B.kind];
    V3 vertsA[8], vertsB[8];
    int na = worldVertsOf(A, vertsA), nb = worldVertsOf(B, vertsB);
    V3 d = sub(B.pos, A.pos);

    // face axes over both bodies
    f64 bestFaceSep = -1e30;
    V3 bestFaceN = v3(0, 0, 0);
    const Face* bestFaceRef = 0;
    const V3* refVerts = 0;
    const V3* incVertsArr = 0;
    const Shape* incShape = 0;
    const Body* incBody = 0;
    bool refIsA = true;

    for (int pass = 0; pass < 2; pass++) {
        const Body& ref = pass == 0 ? A : B;
        const Shape& rs = pass == 0 ? sa : sb;
        const V3* rv = pass == 0 ? vertsA : vertsB;
        const V3* ov = pass == 0 ? vertsB : vertsA;
        int on = pass == 0 ? nb : na;
        for (int f = 0; f < rs.faceCount; f++) {
            V3 n = rotLocal(ref, rs.faces[f].normal);
            f64 faceD = dot(n, rv[rs.faces[f].v[0]]);
            f64 minProj = 1e30;
            for (int v = 0; v < on; v++) minProj = fmind(minProj, dot(n, ov[v]));
            f64 sep = minProj - faceD;
            if (sep > bestFaceSep) {
                bestFaceSep = sep;
                bestFaceN = n;
                bestFaceRef = &rs.faces[f];
                refVerts = rv;
                incVertsArr = ov;
                incShape = pass == 0 ? &sb : &sa;
                incBody = pass == 0 ? &B : &A;
                refIsA = pass == 0;
            }
        }
        if (bestFaceSep > 0) return;   // separating plane found
    }

    // edge-cross axes, oriented A -> B
    f64 bestEdgeSep = -1e30;
    V3 bestEdgeL = v3(0, 0, 0), bestEA = v3(0, 0, 0), bestEB = v3(0, 0, 0);
    for (int i = 0; i < sa.dirCount; i++) {
        V3 ea = rotLocal(A, sa.edgeDirs[i]);
        for (int j = 0; j < sb.dirCount; j++) {
            V3 eb = rotLocal(B, sb.edgeDirs[j]);
            V3 L = cross(ea, eb);
            f64 l2 = len2(L);
            if (l2 < 1e-8) continue;
            L = mul(L, 1.0 / fsqrt(l2));
            if (dot(L, d) < 0) L = neg(L);
            f64 maxA = -1e30, minB = 1e30;
            for (int v = 0; v < na; v++) maxA = fmaxd(maxA, dot(L, vertsA[v]));
            for (int v = 0; v < nb; v++) minB = fmind(minB, dot(L, vertsB[v]));
            f64 sep = minB - maxA;
            if (sep > 0) return;
            if (sep > bestEdgeSep) { bestEdgeSep = sep; bestEdgeL = L; bestEA = ea; bestEB = eb; }
        }
    }

    // Prefer face contacts; only take the edge case when clearly shallower.
    if (-bestEdgeSep < -bestFaceSep * 0.95 - 1e-3) {
        // supporting edge on each body: parallel to the direction, furthest along L
        const Edge* eA = 0; const Edge* eB = 0;
        f64 bestProj = -1e30;
        for (int e = 0; e < sa.edgeCount; e++) {
            V3 ev = sub(vertsA[sa.edges[e].j], vertsA[sa.edges[e].i]);
            if (len2(cross(ev, bestEA)) > 1e-6 * len2(ev)) continue;
            f64 proj = dot(bestEdgeL, vertsA[sa.edges[e].i]) + dot(bestEdgeL, vertsA[sa.edges[e].j]);
            if (proj > bestProj) { bestProj = proj; eA = &sa.edges[e]; }
        }
        bestProj = -1e30;
        for (int e = 0; e < sb.edgeCount; e++) {
            V3 ev = sub(vertsB[sb.edges[e].j], vertsB[sb.edges[e].i]);
            if (len2(cross(ev, bestEB)) > 1e-6 * len2(ev)) continue;
            f64 proj = -(dot(bestEdgeL, vertsB[sb.edges[e].i]) + dot(bestEdgeL, vertsB[sb.edges[e].j]));
            if (proj > bestProj) { bestProj = proj; eB = &sb.edges[e]; }
        }
        if (!eA || !eB) return;
        V3 c1, c2;
        closestSegSeg(vertsA[eA->i], vertsA[eA->j], vertsB[eB->i], vertsB[eB->j], c1, c2);
        pushContact(ia, ib, mul(add(c1, c2), 0.5), bestEdgeL, -bestEdgeSep, E_CUBE, MU_CUBE);
        return;
    }

    V3 refN = bestFaceN;                          // outward from ref, toward inc
    V3 n = refIsA ? refN : neg(refN);             // contact normal A -> B

    // incident face: most anti-parallel to refN
    const Face* incFace = 0;
    f64 most = 1e30;
    for (int f = 0; f < incShape->faceCount; f++) {
        f64 facing = dot(rotLocal(*incBody, incShape->faces[f].normal), refN);
        if (facing < most) { most = facing; incFace = &incShape->faces[f]; }
    }
    V3 polyA[12], polyB[12];
    V3* poly = polyA; V3* tmp = polyB;
    int pn = incFace->n;
    for (int k = 0; k < pn; k++) poly[k] = incVertsArr[incFace->v[k]];

    // clip against the side planes of the reference face
    for (int k = 0; k < bestFaceRef->n && pn; k++) {
        V3 p = refVerts[bestFaceRef->v[k]];
        V3 q = refVerts[bestFaceRef->v[(k + 1) % bestFaceRef->n]];
        V3 side = norm(cross(sub(q, p), refN));
        pn = clipPolyBuf(poly, pn, side, dot(side, p), tmp);
        V3* t = poly; poly = tmp; tmp = t;
    }

    // every clipped point below the reference face becomes a contact
    f64 faceD = dot(refN, refVerts[bestFaceRef->v[0]]);
    int made = 0;
    for (int k = 0; k < pn; k++) {
        f64 sep = dot(refN, poly[k]) - faceD;
        if (sep < 0) { pushContact(ia, ib, poly[k], n, -sep, E_CUBE, MU_CUBE); made++; }
    }
    if (!made) pushContact(ia, ib, mul(add(A.pos, B.pos), 0.5), n, -bestFaceSep, E_CUBE, MU_CUBE);
}

// ===================== collision: sphere cases ======================

static void sphereSphere(int ia, int ib) {
    Body& A = bodies[ia];
    Body& B = bodies[ib];
    V3 d = sub(B.pos, A.pos);
    f64 rsum = shapes[A.kind].r + shapes[B.kind].r;
    f64 d2 = len2(d);
    if (d2 >= rsum * rsum) return;
    f64 dist = fsqrt(d2);
    if (dist < 1e-9) dist = 1e-9;
    V3 n = mul(d, 1.0 / dist);
    V3 p = add(A.pos, mul(n, shapes[A.kind].r - (rsum - dist) / 2));
    pushContact(ia, ib, p, n, rsum - dist, E_CUBE, MU_CUBE);
}

// Sphere vs convex: closest point on the hull to the sphere's center.
// flip=false: sphere is body A; flip=true: sphere is body B (normal A -> B).
static void sphereConvex(int is, int ic, bool flip) {
    Body& S = bodies[is];
    Body& C = bodies[ic];
    const Shape& cs = shapes[C.kind];
    V3 verts[8];
    worldVertsOf(C, verts);
    V3 P = S.pos;
    f64 r = shapes[S.kind].r;

    bool inside = true;
    f64 deepSep = -1e30; V3 deepN = v3(0, 0, 0);
    struct FrontFace { const Face* f; V3 n; f64 sep; };
    FrontFace front[6]; int frontCount = 0;
    for (int f = 0; f < cs.faceCount; f++) {
        V3 n = rotLocal(C, cs.faces[f].normal);
        f64 sep = dot(n, P) - dot(n, verts[cs.faces[f].v[0]]);
        if (sep >= r) return;   // a face plane separates by more than r
        if (sep > 0) { inside = false; front[frontCount++] = { &cs.faces[f], n, sep }; }
        else if (sep > deepSep) { deepSep = sep; deepN = n; }
    }

    V3 q, nrm;
    f64 pen;
    if (inside) {   // center swallowed by the hull: push out through nearest face
        q = sub(P, mul(deepN, deepSep));
        nrm = neg(deepN);
        pen = r - deepSep;
    } else {
        V3 bestQ = v3(0, 0, 0);
        f64 bestD2 = 1e30;
        for (int fi = 0; fi < frontCount; fi++) {
            const Face* f = front[fi].f;
            V3 n = front[fi].n;
            f64 sep = front[fi].sep;
            V3 proj = sub(P, mul(n, sep));
            bool inPoly = true;
            for (int k = 0; k < f->n; k++) {
                V3 p1 = verts[f->v[k]], p2 = verts[f->v[(k + 1) % f->n]];
                if (dot(sub(proj, p1), cross(sub(p2, p1), n)) > 0) { inPoly = false; break; }
            }
            if (inPoly) {
                if (sep * sep < bestD2) { bestD2 = sep * sep; bestQ = proj; }
                continue;
            }
            for (int k = 0; k < f->n; k++) {   // nearest point on the face's edges
                V3 p1 = verts[f->v[k]], p2 = verts[f->v[(k + 1) % f->n]];
                V3 ev = sub(p2, p1);
                f64 t = clampd(dot(sub(P, p1), ev) / len2(ev), 0, 1);
                V3 cand = add(p1, mul(ev, t));
                f64 d2c = len2(sub(P, cand));
                if (d2c < bestD2) { bestD2 = d2c; bestQ = cand; }
            }
        }
        if (bestD2 >= r * r) return;
        f64 dist = fsqrt(bestD2);
        if (dist < 1e-9) dist = 1e-9;
        q = bestQ;
        nrm = mul(sub(bestQ, P), 1.0 / dist);   // sphere -> convex
        pen = r - dist;
    }
    if (flip) pushContact(ic, is, q, neg(nrm), pen, E_CUBE, MU_CUBE);   // convex is A
    else      pushContact(is, ic, q, nrm, pen, E_CUBE, MU_CUBE);        // sphere is A
}

static void collide(int i, int j) {
    bool si = shapes[bodies[i].kind].isSphere, sj = shapes[bodies[j].kind].isSphere;
    if (si && sj) sphereSphere(i, j);
    else if (si) sphereConvex(i, j, false);
    else if (sj) sphereConvex(j, i, true);
    else convexConvex(i, j);
}

// ================ contact solver (sequential impulses) ==============

static inline V3 velAt(const Body& b, V3 r) { return add(b.vel, cross(b.angVel, r)); }
static inline V3 relVel(const Contact& c) {
    V3 vb = velAt(bodies[c.b], c.rb);
    if (c.a < 0) return vb;
    return sub(vb, velAt(bodies[c.a], c.ra));
}
static inline void applyImpulse(Contact& c, V3 P) {
    if (c.imB != 0) {
        Body& b = bodies[c.b];
        b.vel = add(b.vel, mul(P, c.imB));
        b.angVel = add(b.angVel, mul(cross(c.rb, P), c.iiB));
    }
    if (c.a >= 0 && c.imA != 0) {
        Body& a = bodies[c.a];
        a.vel = sub(a.vel, mul(P, c.imA));
        a.angVel = sub(a.angVel, mul(cross(c.ra, P), c.iiA));
    }
}
static inline f64 effMass(const Contact& c, V3 dir) {
    f64 k = c.imB + c.iiB * len2(cross(c.rb, dir));
    if (c.a >= 0) k += c.imA + c.iiA * len2(cross(c.ra, dir));
    return 1.0 / k;
}

static void prepContact(Contact& c, f64 h) {
    const Body& B = bodies[c.b];
    c.imB = B.sleeping ? 0 : 1.0 / CUBE_MASS;
    c.iiB = B.sleeping ? 0 : B.invI;
    if (c.a >= 0) {
        const Body& A = bodies[c.a];
        c.imA = A.sleeping ? 0 : 1.0 / CUBE_MASS;
        c.iiA = A.sleeping ? 0 : A.invI;
        c.ra = sub(c.p, A.pos);
    } else {
        c.imA = 0; c.iiA = 0;
    }
    c.rb = sub(c.p, B.pos);
    c.t1 = norm(fabsd(c.n.x) > 0.9 ? cross(c.n, v3(0, 1, 0)) : cross(c.n, v3(1, 0, 0)));
    c.t2 = cross(c.n, c.t1);
    c.massN  = effMass(c, c.n);
    c.massT1 = effMass(c, c.t1);
    c.massT2 = effMass(c, c.t2);
    f64 vn0 = dot(relVel(c), c.n);
    f64 bias = fmind(MAX_BIAS, (BAUMGARTE / h) * fmaxd(0, c.pen - PEN_SLOP));
    f64 rest = vn0 < -REST_THRESHOLD ? -c.e * vn0 : 0;
    c.target = fmaxd(bias, rest);
    c.accN = 0; c.accT1 = 0; c.accT2 = 0;
}

static inline void clampedImpulse(Contact& c, V3 dir, f64 dLambda, f64& acc, f64 lo, f64 hi) {
    f64 old = acc;
    acc = clampd(old + dLambda, lo, hi);
    f64 applied = acc - old;
    if (applied != 0) applyImpulse(c, mul(dir, applied));
}

static void solveContact(Contact& c) {
    clampedImpulse(c, c.n, (c.target - dot(relVel(c), c.n)) * c.massN, c.accN, 0, 1e30);
    f64 maxF = c.mu * c.accN;
    clampedImpulse(c, c.t1, -dot(relVel(c), c.t1) * c.massT1, c.accT1, -maxF, maxF);
    clampedImpulse(c, c.t2, -dot(relVel(c), c.t2) * c.massT2, c.accT2, -maxF, maxF);
}

// ============================ simulation ===========================

static void removeAt(int i) {   // order-preserving (matches JS splice)
    for (int k = i; k < bodyCount - 1; k++) bodies[k] = bodies[k + 1];
    bodyCount--;
}

EXPORT void step(f64 h) {
    bool anyAwake = false;
    for (int i = 0; i < bodyCount; i++) {
        Body& b = bodies[i];
        b.touch = false;
        if (b.sleeping) continue;
        anyAwake = true;
        b.vel.y += GRAVITY * h;
        b.vel = mul(b.vel, 1 - LIN_DAMP * h);
        b.angVel = mul(b.angVel, 1 - ANG_DAMP * h);
    }
    if (!anyAwake) return;

    buildGrid();
    contactCount = 0;

    // floor contacts (same rules and edge guards as simulation.js)
    for (int i = 0; i < bodyCount; i++) {
        Body& b = bodies[i];
        if (b.sleeping || b.pos.y > b.radius + 0.1) continue;

        if (shapes[b.kind].isSphere) {
            if (b.pos.y <= 0) continue;
            f64 r = shapes[b.kind].r;
            V3 q = v3(clampd(b.pos.x, -FLOOR_HALF, FLOOR_HALF), 0,
                      clampd(b.pos.z, -FLOOR_HALF, FLOOR_HALF));
            V3 dq = sub(b.pos, q);
            f64 d2 = len2(dq);
            if (d2 >= r * r) continue;
            f64 dist = fsqrt(d2);
            if (dist < 1e-9) dist = 1e-9;
            pushContact(-1, i, q, mul(dq, 1.0 / dist), r - dist, E_FLOOR, MU_FLOOR);
            b.touch = true;
            continue;
        }

        V3 verts[8];
        int nv = worldVertsOf(b, verts);
        bool under = false;
        for (int v = 0; v < nv; v++) {
            if (-verts[v].y > 0.3 && fabsd(verts[v].x) <= FLOOR_HALF && fabsd(verts[v].z) <= FLOOR_HALF) {
                under = true;
                break;
            }
        }
        if (under) continue;
        for (int v = 0; v < nv; v++) {
            if (verts[v].y >= 0.02) continue;
            f64 depth = -verts[v].y;
            f64 inset = fmind(FLOOR_HALF - fabsd(verts[v].x), FLOOR_HALF - fabsd(verts[v].z));
            if (inset < 0 || inset < depth) continue;
            pushContact(-1, i, verts[v], v3(0, 1, 0), fmaxd(0, depth), E_FLOOR, MU_FLOOR);
            b.touch = true;
        }
        // rim contacts: pivot on the floor's actual edge (capped penetration)
        if (fabsd(b.pos.x) + b.radius > FLOOR_HALF || fabsd(b.pos.z) + b.radius > FLOOR_HALF) {
            const Shape& s = shapes[b.kind];
            for (int e = 0; e < s.edgeCount; e++) {
                V3 p1 = verts[s.edges[e].i], p2 = verts[s.edges[e].j];
                for (int axis = 0; axis < 2; axis++) {
                    f64 c1 = axis ? p1.z : p1.x, c2 = axis ? p2.z : p2.x;
                    for (int bs = 0; bs < 2; bs++) {
                        f64 bound = bs ? FLOOR_HALF : -FLOOR_HALF;
                        if ((c1 < bound) == (c2 < bound)) continue;
                        f64 t = (bound - c1) / (c2 - c1);
                        f64 y = p1.y + t * (p2.y - p1.y);
                        if (y >= 0.02 || -y > 0.3) continue;
                        f64 other = axis ? p1.x + t * (p2.x - p1.x) : p1.z + t * (p2.z - p1.z);
                        if (fabsd(other) > FLOOR_HALF) continue;
                        V3 p = axis ? v3(other, y, bound) : v3(bound, y, other);
                        pushContact(-1, i, p, v3(0, 1, 0), fmind(fmaxd(0, -y), 0.05), E_FLOOR, MU_FLOOR);
                        b.touch = true;
                    }
                }
            }
        }
    }

    // body-body contacts via the spatial hash; only awake bodies scan
    for (int i = 0; i < bodyCount; i++) {
        Body& A = bodies[i];
        if (A.sleeping) continue;
        forNeighbors(A, [&](int j) {
            if (j == i) return;
            Body& B = bodies[j];
            if (!B.sleeping && j <= i) return;   // visit each awake pair once
            f64 r = A.radius + B.radius;
            if (len2(sub(B.pos, A.pos)) > r * r) return;
            if (B.sleeping && len2(A.vel) > WAKE_SPEED2) wakeChain(B);
            int before = contactCount;
            collide(i, j);
            if (contactCount > before) { A.touch = true; B.touch = true; }
        });
    }

    for (int c = 0; c < contactCount; c++) prepContact(contacts[c], h);
    for (int it = 0; it < SOLVER_ITERS; it++) {
        for (int c = 0; c < contactCount; c++) solveContact(contacts[c]);
    }

    // a solved impulse that exceeds quiet weight transfer is a shove: wake
    for (int c = 0; c < contactCount; c++) {
        if (contacts[c].accN < WAKE_IMPULSE) continue;
        if (contacts[c].a >= 0 && bodies[contacts[c].a].sleeping) wakeChain(bodies[contacts[c].a]);
        if (bodies[contacts[c].b].sleeping) wakeChain(bodies[contacts[c].b]);
    }

    for (int i = 0; i < bodyCount; i++) {
        Body& b = bodies[i];
        if (b.sleeping) continue;
        b.pos = add(b.pos, mul(b.vel, h));
        Quat w = { 0, b.angVel.x, b.angVel.y, b.angVel.z };
        Quat dq = qMul(w, b.q);
        b.q.w += 0.5 * dq.w * h;
        b.q.x += 0.5 * dq.x * h;
        b.q.y += 0.5 * dq.y * h;
        b.q.z += 0.5 * dq.z * h;
        qNormalize(b.q);
        updateAxes(b);
        if (b.touch && shapes[b.kind].isSphere) {   // rolling resistance
            b.angVel = mul(b.angVel, fmaxd(0, 1 - ROLL_RESIST * h));
        }
        if (b.touch && len2(b.vel) < SLEEP_LIN2 && len2(b.angVel) < SLEEP_ANG2) {
            b.vel = mul(b.vel, 0.94);
            b.angVel = mul(b.angVel, 0.94);
            if (++b.sleepTimer >= SLEEP_STEPS) {
                b.sleeping = true;
                b.vel = v3(0, 0, 0);
                b.angVel = v3(0, 0, 0);
            }
        } else {
            b.sleepTimer = b.sleepTimer > 4 ? b.sleepTimer - 4 : 0;
        }
    }

    for (int i = bodyCount - 1; i >= 0; i--) {
        if (bodies[i].pos.y < -40) removeAt(i);   // the JS mirror notices the id vanish
    }
}

// ========================== JS interface ===========================

EXPORT void init() {
    buildShapes();
    bodyCount = 0;
}

EXPORT void clear_scene() {
    bodyCount = 0;
}

EXPORT void add_body(int id, int kind,
                     f64 px, f64 py, f64 pz,
                     f64 qw, f64 qx, f64 qy, f64 qz,
                     f64 vx, f64 vy, f64 vz,
                     f64 wx, f64 wy, f64 wz,
                     int sleeping) {
    if (bodyCount >= MAX_BODIES) return;
    Body& b = bodies[bodyCount++];
    b.id = id;
    b.kind = kind;
    b.pos = v3(px, py, pz);
    b.q = { qw, qx, qy, qz };
    b.vel = v3(vx, vy, vz);
    b.angVel = v3(wx, wy, wz);
    b.invI = shapes[kind].invI;
    b.radius = shapes[kind].radius;
    b.sleeping = sleeping != 0;
    b.sleepTimer = 0;
    b.touch = false;
    updateAxes(b);
}

EXPORT void remove_body(int id) {   // cap eviction: wake what rested on it
    for (int i = 0; i < bodyCount; i++) {
        if (bodies[i].id != id) continue;
        V3 pos = bodies[i].pos;
        f64 rad = bodies[i].radius;
        removeAt(i);
        for (int k = 0; k < bodyCount; k++) {
            f64 rr = bodies[k].radius + rad;
            if (bodies[k].sleeping && len2(sub(bodies[k].pos, pos)) < rr * rr) wakeBody(bodies[k]);
        }
        return;
    }
}

static u32 rngState = 1;
static f64 frand() {   // LCG: same spirit as Math.random, determinism not required
    rngState = rngState * 1664525u + 1013904223u;
    return (f64)(rngState >> 8) / 16777216.0;
}

EXPORT void bounce_all(f64 seed) {
    rngState = (u32)seed | 1;
    for (int i = 0; i < bodyCount; i++) {
        Body& b = bodies[i];
        wakeBody(b);
        b.vel = add(b.vel, v3(0, 5.5 + frand() * 2.5, 0));
        b.angVel = add(b.angVel, v3((frand() * 2 - 1) * 1.5,
                                    (frand() * 2 - 1) * 1.5,
                                    (frand() * 2 - 1) * 1.5));
    }
}

EXPORT int body_count() { return bodyCount; }

// Packed state the JS mirror reads each frame (16 doubles per body):
// [id, kind, pos xyz, quat wxyz, vel xyz, angVel xyz, sleeping]
static f64 stateBuf[MAX_BODIES * 16];

EXPORT f64* state_ptr() {
    for (int i = 0; i < bodyCount; i++) {
        const Body& b = bodies[i];
        f64* o = &stateBuf[i * 16];
        o[0] = (f64)b.id;  o[1] = (f64)b.kind;
        o[2] = b.pos.x;    o[3] = b.pos.y;    o[4] = b.pos.z;
        o[5] = b.q.w;      o[6] = b.q.x;      o[7] = b.q.y;      o[8] = b.q.z;
        o[9] = b.vel.x;    o[10] = b.vel.y;   o[11] = b.vel.z;
        o[12] = b.angVel.x; o[13] = b.angVel.y; o[14] = b.angVel.z;
        o[15] = b.sleeping ? 1.0 : 0.0;
    }
    return stateBuf;
}
