"use strict";
// Fable 5 Physics Sandbox — written by Claude Fable 5 (Anthropic).
// MIT License; see LICENSE and README.md.
// ===========================================================================
// WEBGPU RENDERER — the modern explicit GPU API.
//
// Same strategy as the WebGL renderer (read that file first — the two are
// deliberately structured alike so they can be compared side by side): one
// instanced draw per shape of a shared mesh, quaternion rotation in the
// vertex shader, flat Lambert shading for the polyhedra, a procedural
// beach-ball pattern with per-fragment lighting for the spheres, planar
// shadows masked by the stencil buffer, and the depth buffer resolving all
// occlusion.
//
// What changes is the shape of the API. WebGL is a state machine you mutate
// between draws (enable blending, set stencil ops, ...). WebGPU front-loads
// all of that into immutable *pipeline* objects created once at startup;
// each frame just records "set pipeline, set buffers, draw" into a command
// encoder and submits it. Shaders are WGSL instead of GLSL, device access is
// asynchronous, and clip-space z spans [0, 1] rather than WebGL's [-1, 1]
// (see the projection matrix below). Rendering is 4x multisampled and
// resolved to the canvas each frame.
// ===========================================================================

const WebGPURenderer = (() => {
    let device = null, context = null, format;
    let convexPipe, spherePipe, shadowPipe, floorPipe, gridPipe;
    let uniformBuf, bindGroup;
    let floorBuf, gridBuf, gridVertCount = 0;
    let shapes = null;    // per-shape mesh + instance buffers (cube/sphere/tet)
    let msaaTex = null, depthTex = null;
    const uniforms = new Float32Array(24);   // mat4 (16) + light vec4 + params vec4

    const SHADER = /* wgsl */`
    struct Uniforms {
        viewProj : mat4x4f,
        light    : vec4f,   // xyz light direction, w ambient
        params   : vec4f    // x diffuse
    };
    @group(0) @binding(0) var<uniform> U : Uniforms;

    // v' = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)
    fn qrotate(q : vec4f, v : vec3f) -> vec3f {
        return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
    }

    struct ConvexOut {
        @builtin(position) pos : vec4f,
        @location(0) color : vec3f
    };

    @vertex fn convex_vs(
        @location(0) aPos : vec3f, @location(1) aNormal : vec3f,
        @location(2) iPos : vec3f, @location(3) iQuat : vec4f, @location(4) iColor : vec3f
    ) -> ConvexOut {
        var out : ConvexOut;
        let world = iPos + qrotate(iQuat, aPos);
        let n = qrotate(iQuat, aNormal);
        let shade = U.light.w + U.params.x * max(dot(n, U.light.xyz), 0.0);
        out.color = iColor * shade;
        out.pos = U.viewProj * vec4f(world, 1.0);
        return out;
    }
    @fragment fn convex_fs(in : ConvexOut) -> @location(0) vec4f {
        return vec4f(in.color, 1.0);
    }

    struct SphereOut {
        @builtin(position) pos : vec4f,
        @location(0) normal : vec3f,
        @location(1) local : vec3f,   // pattern lives in the sphere's own frame
        @location(2) base : vec3f
    };

    @vertex fn sphere_vs(
        @location(0) aPos : vec3f, @location(1) aNormal : vec3f,
        @location(2) iPos : vec3f, @location(3) iQuat : vec4f, @location(4) iColor : vec3f
    ) -> SphereOut {
        var out : SphereOut;
        let world = iPos + qrotate(iQuat, aPos);
        out.normal = qrotate(iQuat, aNormal);
        out.local = aPos;
        out.base = iColor;
        out.pos = U.viewProj * vec4f(world, 1.0);
        return out;
    }
    @fragment fn sphere_fs(in : SphereOut) -> @location(0) vec4f {
        // six beach-ball wedges around the sphere's own polar axis
        let wedge = floor((atan2(in.local.z, in.local.x) / 6.28318530718 + 0.5) * 6.0);
        let base = select(vec3f(0.95), in.base, (wedge % 2.0) < 0.5);
        let shade = U.light.w + U.params.x * max(dot(normalize(in.normal), U.light.xyz), 0.0);
        return vec4f(base * shade, 1.0);
    }

    struct ShadowOut {
        @builtin(position) pos : vec4f,
        @location(0) alpha : f32
    };

    @vertex fn shadow_vs(
        @location(0) aPos : vec3f,
        @location(2) iPos : vec3f, @location(3) iQuat : vec4f, @location(5) iAlpha : f32
    ) -> ShadowOut {
        var out : ShadowOut;
        var world = iPos + qrotate(iQuat, aPos);
        world.y = 0.006;                    // squash onto the floor plane
        out.alpha = iAlpha;
        out.pos = U.viewProj * vec4f(world, 1.0);
        return out;
    }
    @fragment fn shadow_fs(in : ShadowOut) -> @location(0) vec4f {
        return vec4f(0.0, 0.0, 0.0, in.alpha);
    }

    @vertex fn flat_vs(@location(0) aPos : vec3f) -> @builtin(position) vec4f {
        return U.viewProj * vec4f(aPos, 1.0);
    }
    @fragment fn floor_fs() -> @location(0) vec4f {
        return vec4f(0.49, 0.49, 0.49, 1.0);
    }
    @fragment fn grid_fs() -> @location(0) vec4f {
        return vec4f(0.0, 0.0, 0.0, 0.15);
    }`;

    // vertex-buffer layouts (matching the WebGL attribute wiring)
    const MESH_LAYOUT = {
        arrayStride: 24,
        attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },    // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }    // normal
        ]
    };
    const INST_LAYOUT = {
        arrayStride: 40,
        stepMode: 'instance',
        attributes: [
            { shaderLocation: 2, offset: 0, format: 'float32x3' },    // world position
            { shaderLocation: 3, offset: 12, format: 'float32x4' },   // quaternion
            { shaderLocation: 4, offset: 28, format: 'float32x3' }    // color
        ]
    };
    const SHADOW_LAYOUT = {
        arrayStride: 32,
        stepMode: 'instance',
        attributes: [
            { shaderLocation: 2, offset: 0, format: 'float32x3' },
            { shaderLocation: 3, offset: 12, format: 'float32x4' },
            { shaderLocation: 5, offset: 28, format: 'float32' }      // alpha
        ]
    };
    const POS_LAYOUT = {
        arrayStride: 12,
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
    };

    const ALPHA_BLEND = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
    };

    // MSAA color target and depth-stencil buffer, recreated on resize.
    function createTargets() {
        msaaTex?.destroy();
        depthTex?.destroy();
        msaaTex = device.createTexture({
            size: [W, H], sampleCount: 4, format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        depthTex = device.createTexture({
            size: [W, H], sampleCount: 4, format: 'depth24plus-stencil8',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }

    // Same view matrix as WebGL, but the projection maps view depth to clip
    // z in [0, 1] — WebGPU's convention — instead of WebGL's [-1, 1].
    function viewProjMatrix(out) {
        const f = 1 / Math.tan(FOV / 2), aspect = W / H;
        const near = 0.1, far = 200;
        const a = f / aspect, b = f;
        const c = far / (near - far), d = far * near / (near - far);
        const zx = -camF.x, zy = -camF.y, zz = -camF.z;   // camera backward axis
        const ex = camPos.x, ey = camPos.y, ez = camPos.z;
        out[0] = a * camR.x; out[4] = a * camR.y; out[8]  = a * camR.z; out[12] = -a * (camR.x * ex + camR.y * ey + camR.z * ez);
        out[1] = b * camU.x; out[5] = b * camU.y; out[9]  = b * camU.z; out[13] = -b * (camU.x * ex + camU.y * ey + camU.z * ez);
        out[2] = c * zx;     out[6] = c * zy;     out[10] = c * zz;     out[14] = -c * (zx * ex + zy * ey + zz * ez) + d;
        out[3] = -zx;        out[7] = -zy;        out[11] = -zz;        out[15] = zx * ex + zy * ey + zz * ez;
    }

    // ========================= renderer interface =======================

    return {
        name: 'WebGPU',

        async init(cv) {
            if (!navigator.gpu) return false;
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return false;
            device = await adapter.requestDevice();
            context = cv.getContext('webgpu');
            format = navigator.gpu.getPreferredCanvasFormat();
            context.configure({ device, format, alphaMode: 'opaque' });

            const module = device.createShaderModule({ code: SHADER });

            // one small uniform buffer shared by every pipeline
            uniformBuf = device.createBuffer({
                size: uniforms.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const bindLayout = device.createBindGroupLayout({
                entries: [{
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                }]
            });
            bindGroup = device.createBindGroup({
                layout: bindLayout,
                entries: [{ binding: 0, resource: { buffer: uniformBuf } }]
            });
            const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });

            // All render state lives in immutable pipelines, created once.
            const makePipe = (opts) => device.createRenderPipeline({
                layout: pipeLayout,
                vertex: { module, entryPoint: opts.vs, buffers: opts.buffers },
                fragment: {
                    module, entryPoint: opts.fs,
                    targets: [{ format, blend: opts.blend }]
                },
                primitive: { topology: opts.topology, cullMode: opts.cull || 'none' },
                depthStencil: {
                    format: 'depth24plus-stencil8',
                    depthWriteEnabled: opts.depthWrite,
                    depthCompare: 'less',
                    stencilFront: opts.stencil, stencilBack: opts.stencil
                },
                multisample: { count: 4 }
            });

            floorPipe = makePipe({
                vs: 'flat_vs', fs: 'floor_fs', buffers: [POS_LAYOUT],
                topology: 'triangle-strip', depthWrite: true
            });
            gridPipe = makePipe({
                vs: 'flat_vs', fs: 'grid_fs', buffers: [POS_LAYOUT],
                topology: 'line-list', depthWrite: false, blend: ALPHA_BLEND
            });
            shadowPipe = makePipe({
                vs: 'shadow_vs', fs: 'shadow_fs', buffers: [MESH_LAYOUT, SHADOW_LAYOUT],
                topology: 'triangle-list', depthWrite: false, blend: ALPHA_BLEND,
                // darken each pixel once: pass only where stencil is still 0,
                // and increment it so overlapping shadow triangles fail
                stencil: { compare: 'equal', passOp: 'increment-clamp' }
            });
            convexPipe = makePipe({
                vs: 'convex_vs', fs: 'convex_fs', buffers: [MESH_LAYOUT, INST_LAYOUT],
                topology: 'triangle-list', depthWrite: true, cull: 'back'
            });
            spherePipe = makePipe({
                vs: 'sphere_vs', fs: 'sphere_fs', buffers: [MESH_LAYOUT, INST_LAYOUT],
                topology: 'triangle-list', depthWrite: true   // closed mesh: skip culling
            });

            // static geometry + per-shape instance buffers
            const upload = (data, usage) => {
                const buf = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
                device.queue.writeBuffer(buf, 0, data);
                return buf;
            };
            const setupShape = (mesh) => ({
                idxCount: mesh.indices.length,
                meshBuf: upload(mesh.verts, GPUBufferUsage.VERTEX),
                idxBuf: upload(mesh.indices, GPUBufferUsage.INDEX),
                instBuf: device.createBuffer({ size: MAX_CUBES * 40, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }),
                shadowBuf: device.createBuffer({ size: MAX_CUBES * 32, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }),
                instData: new Float32Array(MAX_CUBES * 10),
                shadowData: new Float32Array(MAX_CUBES * 8),
                n: 0, s: 0
            });
            shapes = {
                cube: setupShape(buildCubeMesh()),
                sphere: setupShape(buildSphereMesh()),
                tet: setupShape(buildTetMesh())
            };
            const F = FLOOR_HALF;   // triangle-strip order
            floorBuf = upload(new Float32Array([-F, 0, -F,  F, 0, -F,  -F, 0, F,  F, 0, F]), GPUBufferUsage.VERTEX);
            const grid = buildGridLines();
            gridVertCount = grid.length / 3;
            gridBuf = upload(grid, GPUBufferUsage.VERTEX);

            createTargets();
            return true;
        },

        resize() {
            if (device) createTargets();
        },

        dispose() {
            device?.destroy();
            device = null;
        },

        render() {
            // per-frame uniforms
            viewProjMatrix(uniforms);
            uniforms[16] = LIGHT_DIR.x; uniforms[17] = LIGHT_DIR.y;
            uniforms[18] = LIGHT_DIR.z; uniforms[19] = AMBIENT;
            uniforms[20] = DIFFUSE;
            device.queue.writeBuffer(uniformBuf, 0, uniforms);

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
            for (const key in shapes) {
                const sh = shapes[key];
                if (sh.n) device.queue.writeBuffer(sh.instBuf, 0, sh.instData, 0, sh.n);
                if (sh.s) device.queue.writeBuffer(sh.shadowBuf, 0, sh.shadowData, 0, sh.s);
            }

            // record the frame: draw into the MSAA target, resolve to canvas
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: msaaTex.createView(),
                    resolveTarget: context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'discard'
                }],
                depthStencilAttachment: {
                    view: depthTex.createView(),
                    depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard',
                    stencilClearValue: 0, stencilLoadOp: 'clear', stencilStoreOp: 'discard'
                }
            });
            pass.setBindGroup(0, bindGroup);

            pass.setPipeline(floorPipe);
            pass.setVertexBuffer(0, floorBuf);
            pass.draw(4);

            pass.setPipeline(gridPipe);
            pass.setVertexBuffer(0, gridBuf);
            pass.draw(gridVertCount);

            pass.setPipeline(shadowPipe);
            pass.setStencilReference(0);
            for (const key in shapes) {
                const sh = shapes[key];
                if (!sh.s) continue;
                pass.setVertexBuffer(0, sh.meshBuf);
                pass.setVertexBuffer(1, sh.shadowBuf);
                pass.setIndexBuffer(sh.idxBuf, 'uint16');
                pass.drawIndexed(sh.idxCount, sh.s / 8);
            }

            const drawShape = (pipe, sh) => {
                if (!sh.n) return;
                pass.setPipeline(pipe);
                pass.setVertexBuffer(0, sh.meshBuf);
                pass.setVertexBuffer(1, sh.instBuf);
                pass.setIndexBuffer(sh.idxBuf, 'uint16');
                pass.drawIndexed(sh.idxCount, sh.n / 10);
            };
            drawShape(convexPipe, shapes.cube);
            drawShape(convexPipe, shapes.tet);
            drawShape(spherePipe, shapes.sphere);

            pass.end();
            device.queue.submit([encoder.finish()]);
        },

        // cache hooks: nothing is cached here — the GPU redraws everything
        bodyWoke() {},
        bodyRemoved() {},
        sceneCleared() {}
    };
})();
