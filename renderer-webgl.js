"use strict";
// Fable 5 Physics Sandbox — written by Claude Fable 5 (Anthropic).
// MIT License; see LICENSE and README.md.
// ===========================================================================
// WEBGL RENDERER — the classic hardware pipeline (WebGL 2).
//
// Where the software renderer must scheme to avoid filling pixels (see
// renderer-software.js), the GPU makes brute force the simple AND fast
// answer: every frame we upload each body's position + orientation
// quaternion + color (10 floats per body) and issue one instanced draw call
// per shape — one for all cubes, one for all tetrahedra, one for all
// spheres. The vertex shader rotates each vertex by the instance quaternion
// and the depth buffer resolves all occlusion per pixel — no sorting, no
// caching, no dirty rectangles.
//
// The flat-shaded polyhedra compute their Lambert term in the vertex shader
// (every vertex of a face shares its normal). Spheres get their own shader
// pair: smooth per-fragment lighting, plus a procedural beach-ball pattern
// derived from the hit point in the sphere's OWN frame — no texture, and the
// pattern tumbles with the body so rolling is visible.
//
// Draw order each frame: floor quad -> grid lines -> shadows -> bodies.
// Shadows are the textbook "planar projection" trick: each mesh is drawn
// again squashed flat onto the floor as translucent black, with the stencil
// buffer ensuring a pixel darkens only once however many triangles overlap.
// ===========================================================================

const WebGLRenderer = (() => {
    let gl = null;
    let convexProg, sphereProg, shadowProg, flatProg;   // the four shader programs
    let floorVAO, gridVAO, gridVerts = 0;
    let shapes = null;    // per-shape mesh + instance buffers (cube/sphere/tet)
    let uni = {};         // uniform locations
    const vp = new Float32Array(16);   // view-projection matrix

    // ============================== shaders =============================

    // Rotating a vector by a quaternion directly in the shader:
    // v' = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)
    const QROTATE = `
    vec3 qrotate(vec4 q, vec3 v) {
        return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
    }`;

    const CONVEX_VS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 aPos;      // mesh vertex
    layout(location = 1) in vec3 aNormal;   // face normal (flat shading)
    layout(location = 2) in vec3 iPos;      // instance: world position
    layout(location = 3) in vec4 iQuat;     // instance: orientation
    layout(location = 4) in vec3 iColor;    // instance: base color
    uniform mat4 uViewProj;
    uniform vec4 uLight;                    // xyz light direction, w ambient
    uniform float uDiffuse;
    out vec3 vColor;
    ${QROTATE}
    void main() {
        vec3 world = iPos + qrotate(iQuat, aPos);
        vec3 n = qrotate(iQuat, aNormal);
        float shade = uLight.w + uDiffuse * max(dot(n, uLight.xyz), 0.0);
        vColor = iColor * shade;
        gl_Position = uViewProj * vec4(world, 1.0);
    }`;

    const CONVEX_FS = `#version 300 es
    precision highp float;
    in vec3 vColor;
    out vec4 outColor;
    void main() { outColor = vec4(vColor, 1.0); }`;

    const SPHERE_VS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aNormal;
    layout(location = 2) in vec3 iPos;
    layout(location = 3) in vec4 iQuat;
    layout(location = 4) in vec3 iColor;
    uniform mat4 uViewProj;
    out vec3 vNormal;
    out vec3 vLocal;
    out vec3 vBase;
    ${QROTATE}
    void main() {
        vec3 world = iPos + qrotate(iQuat, aPos);
        vNormal = qrotate(iQuat, aNormal);
        vLocal = aPos;      // pattern lives in the sphere's own frame: it tumbles
        vBase = iColor;
        gl_Position = uViewProj * vec4(world, 1.0);
    }`;

    const SPHERE_FS = `#version 300 es
    precision highp float;
    in vec3 vNormal;
    in vec3 vLocal;
    in vec3 vBase;
    uniform vec4 uLight;
    uniform float uDiffuse;
    out vec4 outColor;
    void main() {
        // six beach-ball wedges around the sphere's own polar axis
        float wedge = floor((atan(vLocal.z, vLocal.x) / 6.28318530718 + 0.5) * 6.0);
        vec3 base = mod(wedge, 2.0) < 0.5 ? vBase : vec3(0.95);
        float shade = uLight.w + uDiffuse * max(dot(normalize(vNormal), uLight.xyz), 0.0);
        outColor = vec4(base * shade, 1.0);
    }`;

    // Shadow: same mesh, same instance transform, but every vertex is
    // squashed onto the floor plane — a straight-down planar projection.
    const SHADOW_VS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 aPos;
    layout(location = 2) in vec3 iPos;
    layout(location = 3) in vec4 iQuat;
    layout(location = 5) in float iAlpha;   // height-faded shadow strength
    uniform mat4 uViewProj;
    out float vAlpha;
    ${QROTATE}
    void main() {
        vec3 world = iPos + qrotate(iQuat, aPos);
        world.y = 0.006;                    // just above the floor: no z-fight
        vAlpha = iAlpha;
        gl_Position = uViewProj * vec4(world, 1.0);
    }`;

    const SHADOW_FS = `#version 300 es
    precision highp float;
    in float vAlpha;
    out vec4 outColor;
    void main() { outColor = vec4(0.0, 0.0, 0.0, vAlpha); }`;

    // Floor and grid: plain solid-color geometry.
    const FLAT_VS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 aPos;
    uniform mat4 uViewProj;
    void main() { gl_Position = uViewProj * vec4(aPos, 1.0); }`;

    const FLAT_FS = `#version 300 es
    precision highp float;
    uniform vec4 uColor;
    out vec4 outColor;
    void main() { outColor = uColor; }`;

    function compile(vsSrc, fsSrc) {
        const shader = (type, src) => {
            const s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
                throw new Error('shader: ' + gl.getShaderInfoLog(s));
            }
            return s;
        };
        const p = gl.createProgram();
        gl.attachShader(p, shader(gl.VERTEX_SHADER, vsSrc));
        gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fsSrc));
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            throw new Error('link: ' + gl.getProgramInfoLog(p));
        }
        return p;
    }

    // ============================== camera ==============================

    // Column-major view-projection matrix from the shared camera basis.
    // View: express points in (camR, camU, -camF) axes. Projection: standard
    // GL perspective mapping view depth to clip z in [-1, 1] — compare with
    // the WebGPU renderer, whose clip z spans [0, 1].
    function viewProjMatrix(out) {
        const f = 1 / Math.tan(FOV / 2), aspect = W / H;
        const near = 0.1, far = 200;
        const a = f / aspect, b = f;
        const c = (far + near) / (near - far), d = 2 * far * near / (near - far);
        const zx = -camF.x, zy = -camF.y, zz = -camF.z;   // camera backward axis
        const ex = camPos.x, ey = camPos.y, ez = camPos.z;
        out[0] = a * camR.x; out[4] = a * camR.y; out[8]  = a * camR.z; out[12] = -a * (camR.x * ex + camR.y * ey + camR.z * ez);
        out[1] = b * camU.x; out[5] = b * camU.y; out[9]  = b * camU.z; out[13] = -b * (camU.x * ex + camU.y * ey + camU.z * ez);
        out[2] = c * zx;     out[6] = c * zy;     out[10] = c * zz;     out[14] = -c * (zx * ex + zy * ey + zz * ez) + d;
        out[3] = -zx;        out[7] = -zy;        out[11] = -zz;        out[15] = zx * ex + zy * ey + zz * ez;
    }

    // ======================== per-shape buffers =========================

    // Each shape gets its mesh on the GPU plus two dynamic instance buffers
    // (bodies and shadows) with matching CPU staging arrays.
    function setupShape(mesh) {
        const meshBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, meshBuf);
        gl.bufferData(gl.ARRAY_BUFFER, mesh.verts, gl.STATIC_DRAW);
        const idxBuf = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);

        const instData = new Float32Array(MAX_CUBES * 10);    // pos3 quat4 color3
        const shadowData = new Float32Array(MAX_CUBES * 8);   // pos3 quat4 alpha1
        const instBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
        gl.bufferData(gl.ARRAY_BUFFER, instData.byteLength, gl.DYNAMIC_DRAW);
        const shadowBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, shadowBuf);
        gl.bufferData(gl.ARRAY_BUFFER, shadowData.byteLength, gl.DYNAMIC_DRAW);

        const meshAttribs = () => {
            gl.bindBuffer(gl.ARRAY_BUFFER, meshBuf);
            gl.enableVertexAttribArray(0);
            gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
            gl.enableVertexAttribArray(1);
            gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        };
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        meshAttribs();
        gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 40, 0);
        gl.vertexAttribDivisor(2, 1);   // divisor 1 = advance per instance
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 40, 12);
        gl.vertexAttribDivisor(3, 1);
        gl.enableVertexAttribArray(4);
        gl.vertexAttribPointer(4, 3, gl.FLOAT, false, 40, 28);
        gl.vertexAttribDivisor(4, 1);

        const shadowVao = gl.createVertexArray();
        gl.bindVertexArray(shadowVao);
        meshAttribs();
        gl.bindBuffer(gl.ARRAY_BUFFER, shadowBuf);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 32, 0);
        gl.vertexAttribDivisor(2, 1);
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 32, 12);
        gl.vertexAttribDivisor(3, 1);
        gl.enableVertexAttribArray(5);
        gl.vertexAttribPointer(5, 1, gl.FLOAT, false, 32, 28);
        gl.vertexAttribDivisor(5, 1);
        gl.bindVertexArray(null);

        return {
            idxCount: mesh.indices.length,
            vao, shadowVao, instBuf, shadowBuf, instData, shadowData,
            n: 0, s: 0   // staged floats this frame
        };
    }

    // ========================= renderer interface =======================

    return {
        name: 'WebGL',

        init(cv) {
            gl = cv.getContext('webgl2', { antialias: true, stencil: true, alpha: false });
            if (!gl) return false;

            convexProg = compile(CONVEX_VS, CONVEX_FS);
            sphereProg = compile(SPHERE_VS, SPHERE_FS);
            shadowProg = compile(SHADOW_VS, SHADOW_FS);
            flatProg = compile(FLAT_VS, FLAT_FS);
            uni = {
                convexVP: gl.getUniformLocation(convexProg, 'uViewProj'),
                convexLight: gl.getUniformLocation(convexProg, 'uLight'),
                convexDiffuse: gl.getUniformLocation(convexProg, 'uDiffuse'),
                sphereVP: gl.getUniformLocation(sphereProg, 'uViewProj'),
                sphereLight: gl.getUniformLocation(sphereProg, 'uLight'),
                sphereDiffuse: gl.getUniformLocation(sphereProg, 'uDiffuse'),
                shadowVP: gl.getUniformLocation(shadowProg, 'uViewProj'),
                flatVP: gl.getUniformLocation(flatProg, 'uViewProj'),
                flatColor: gl.getUniformLocation(flatProg, 'uColor')
            };

            shapes = {
                cube: setupShape(buildCubeMesh()),
                sphere: setupShape(buildSphereMesh()),
                tet: setupShape(buildTetMesh())
            };

            const F = FLOOR_HALF;
            const floorBuf = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, floorBuf);
            gl.bufferData(gl.ARRAY_BUFFER,
                new Float32Array([-F, 0, -F,  F, 0, -F,  F, 0, F,  -F, 0, F]), gl.STATIC_DRAW);
            floorVAO = gl.createVertexArray();
            gl.bindVertexArray(floorVAO);
            gl.bindBuffer(gl.ARRAY_BUFFER, floorBuf);
            gl.enableVertexAttribArray(0);
            gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);

            const grid = buildGridLines();
            gridVerts = grid.length / 3;
            const gridBuf = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
            gl.bufferData(gl.ARRAY_BUFFER, grid, gl.STATIC_DRAW);
            gridVAO = gl.createVertexArray();
            gl.bindVertexArray(gridVAO);
            gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
            gl.enableVertexAttribArray(0);
            gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
            gl.bindVertexArray(null);

            gl.clearColor(0, 0, 0, 1);
            this.resize();
            return true;
        },

        resize() {
            if (gl) gl.viewport(0, 0, W, H);
        },

        dispose() {
            gl?.getExtension('WEBGL_lose_context')?.loseContext();
            gl = null;
        },

        render() {
            viewProjMatrix(vp);

            // stage every body's instance data, grouped by shape
            for (const key in shapes) { shapes[key].n = 0; shapes[key].s = 0; }
            for (const b of bodies) {
                const sh = shapes[b.shape.name];
                let n = sh.n;
                sh.instData[n++] = b.pos.x; sh.instData[n++] = b.pos.y; sh.instData[n++] = b.pos.z;
                sh.instData[n++] = b.q.x; sh.instData[n++] = b.q.y; sh.instData[n++] = b.q.z; sh.instData[n++] = b.q.w;
                sh.instData[n++] = b.color.r / 255; sh.instData[n++] = b.color.g / 255; sh.instData[n++] = b.color.b / 255;
                sh.n = n;
                if (b.pos.y < -0.5 || Math.abs(b.pos.x) > FLOOR_HALF || Math.abs(b.pos.z) > FLOOR_HALF) continue;
                let s = sh.s;
                sh.shadowData[s++] = b.pos.x; sh.shadowData[s++] = b.pos.y; sh.shadowData[s++] = b.pos.z;
                sh.shadowData[s++] = b.q.x; sh.shadowData[s++] = b.q.y; sh.shadowData[s++] = b.q.z; sh.shadowData[s++] = b.q.w;
                sh.shadowData[s++] = 0.35 * Math.max(0.08, 1 - b.pos.y / 12);
                sh.s = s;
            }

            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);

            // floor
            gl.useProgram(flatProg);
            gl.uniformMatrix4fv(uni.flatVP, false, vp);
            gl.uniform4f(uni.flatColor, 0.49, 0.49, 0.49, 1);
            gl.bindVertexArray(floorVAO);
            gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

            // grid: translucent lines sitting just above the floor
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            gl.depthMask(false);
            gl.uniform4f(uni.flatColor, 0, 0, 0, 0.15);
            gl.bindVertexArray(gridVAO);
            gl.drawArrays(gl.LINES, 0, gridVerts);

            // shadows: squashed instances of every mesh; the stencil buffer
            // lets each pixel darken once even where triangles overlap
            gl.useProgram(shadowProg);
            gl.uniformMatrix4fv(uni.shadowVP, false, vp);
            gl.enable(gl.STENCIL_TEST);
            gl.stencilFunc(gl.EQUAL, 0, 0xff);         // only untouched pixels
            gl.stencilOp(gl.KEEP, gl.KEEP, gl.INCR);   // and mark them written
            for (const key in shapes) {
                const sh = shapes[key];
                if (!sh.s) continue;
                gl.bindBuffer(gl.ARRAY_BUFFER, sh.shadowBuf);
                gl.bufferSubData(gl.ARRAY_BUFFER, 0, sh.shadowData.subarray(0, sh.s));
                gl.bindVertexArray(sh.shadowVao);
                gl.drawElementsInstanced(gl.TRIANGLES, sh.idxCount, gl.UNSIGNED_SHORT, 0, sh.s / 8);
            }
            gl.disable(gl.STENCIL_TEST);
            gl.depthMask(true);
            gl.disable(gl.BLEND);

            // bodies: one instanced draw per shape
            const drawShape = (prog, sh, cull) => {
                if (!sh.n) return;
                gl.bindBuffer(gl.ARRAY_BUFFER, sh.instBuf);
                gl.bufferSubData(gl.ARRAY_BUFFER, 0, sh.instData.subarray(0, sh.n));
                gl.useProgram(prog);
                if (cull) gl.enable(gl.CULL_FACE); else gl.disable(gl.CULL_FACE);
                gl.bindVertexArray(sh.vao);
                gl.drawElementsInstanced(gl.TRIANGLES, sh.idxCount, gl.UNSIGNED_SHORT, 0, sh.n / 10);
            };
            gl.useProgram(convexProg);
            gl.uniformMatrix4fv(uni.convexVP, false, vp);
            gl.uniform4f(uni.convexLight, LIGHT_DIR.x, LIGHT_DIR.y, LIGHT_DIR.z, AMBIENT);
            gl.uniform1f(uni.convexDiffuse, DIFFUSE);
            drawShape(convexProg, shapes.cube, true);
            drawShape(convexProg, shapes.tet, true);
            gl.useProgram(sphereProg);
            gl.uniformMatrix4fv(uni.sphereVP, false, vp);
            gl.uniform4f(uni.sphereLight, LIGHT_DIR.x, LIGHT_DIR.y, LIGHT_DIR.z, AMBIENT);
            gl.uniform1f(uni.sphereDiffuse, DIFFUSE);
            drawShape(sphereProg, shapes.sphere, false);   // closed mesh: skip culling
            gl.disable(gl.CULL_FACE);
            gl.bindVertexArray(null);
        },

        // cache hooks: nothing is cached here — the GPU redraws everything
        bodyWoke() {},
        bodyRemoved() {},
        sceneCleared() {}
    };
})();
