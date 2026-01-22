#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include <imgui.h>
#include <iostream>
#include <vector>
#include <fstream>

#define CORE_IMPLEMENTATION
#define MATH_IMPLEMENTATION
#define KEYS_IMPLEMENTATION
#define CAMERA_IMPLEMENTATION
#define MODEL_IMPLEMENTATION
#define SDL_IMPLEMENTATION
#define IMGUI_IMPLEMENTATION
#include <wrapper/core.h>

#define BVH_IMPLEMENTATION
#include "bvh.h"

#define CUBE_GRID   32
#define CUBE_SIZE   1.0f
#define CUBE_PAD    0.25f
#define PATH "../res/cube.obj"
#define WIDTH 800
#define HEIGHT 600
#define RENDER_SCALE 0.5f
#define MAX_MODELS CUBE_GRID*CUBE_GRID*CUBE_GRID

struct GPUTriangle { float v0[3], pad0, v1[3], pad1, v2[3], pad2, color[3], pad3; };
struct GPUBVHNode { float bounds_min[3], pad0, bounds_max[3], pad1; int left_idx, right_idx, tri_offset, tri_count; };
struct CameraUniforms { float position[3], pad0, forward[3], pad1, right[3], pad2, up[3], pad3; float vp_width, vp_height; int width, height; };

typedef struct {
    Window_t win;
    SDL_Texture* texture;
    SDL_GPUDevice* gpu_device;
    SDL_GPUComputePipeline* compute_pipeline;
    SDL_GPUBuffer *triangle_buffer, *bvh_buffer, *camera_buffer;
    SDL_GPUTexture* output_texture;
    Camera cam;
    Input input;
    Model models[MAX_MODELS];
    int num_models;
    std::vector<GPUTriangle> gpu_triangles;
    std::vector<GPUBVHNode> gpu_bvh_nodes;
    BVHNode* bvh_root;
    bool running;
    float move_speed, mouse_sensitivity;
} State;

State state = {};

#define ASSERT(x) do { if(!(x)) { std::cout << "Error: " << #x << " " << SDL_GetError() << std::endl; cleanup(); exit(1); } } while(0)
#define LOG(x) do { std::cout << x << std::endl; } while(0)

void cleanup() {
    if (state.bvh_root) bvh_free(state.bvh_root);
    for (int i = 0; i < state.num_models; i++) modelFree(&state.models[i]);
    if (state.compute_pipeline) SDL_ReleaseGPUComputePipeline(state.gpu_device, state.compute_pipeline);
    if (state.triangle_buffer) SDL_ReleaseGPUBuffer(state.gpu_device, state.triangle_buffer);
    if (state.bvh_buffer) SDL_ReleaseGPUBuffer(state.gpu_device, state.bvh_buffer);
    if (state.camera_buffer) SDL_ReleaseGPUBuffer(state.gpu_device, state.camera_buffer);
    if (state.output_texture) SDL_ReleaseGPUTexture(state.gpu_device, state.output_texture);
    if (state.gpu_device) SDL_DestroyGPUDevice(state.gpu_device);
    if (state.texture) SDL_DestroyTexture(state.texture);
    destroyWindow(&state.win);
}

void flatten_bvh(const BVHNode* node, std::vector<GPUBVHNode>& nodes, int& idx, int& tri_off) {
    if (!node) return;
    const int my_idx = idx++;
    GPUBVHNode gn = {};
    gn.bounds_min[0] = node->bounds.min.x; gn.bounds_min[1] = node->bounds.min.y; gn.bounds_min[2] = node->bounds.min.z;
    gn.bounds_max[0] = node->bounds.max.x; gn.bounds_max[1] = node->bounds.max.y; gn.bounds_max[2] = node->bounds.max.z;
    if (node->count > 0) {
        gn.left_idx = gn.right_idx = -1;
        gn.tri_offset = tri_off;
        gn.tri_count = node->count;
        tri_off += node->count;
        nodes.push_back(gn);
    } else {
        gn.tri_offset = -1; gn.tri_count = 0;
        gn.left_idx = idx;
        nodes.push_back(gn);
        flatten_bvh(node->left, nodes, idx, tri_off);
        nodes[my_idx].right_idx = idx;
        flatten_bvh(node->right, nodes, idx, tri_off);
    }
}

std::vector<uint8_t> load_shader(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { LOG("Failed: " << path); return {}; }
    const size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

void init_gpu() {
    state.gpu_device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_MSL, true, nullptr);
    ASSERT(state.gpu_device);

    const auto code = load_shader("../shaders/raytrace.comp.metal");
    ASSERT(!code.empty());

    SDL_GPUComputePipelineCreateInfo pi{};
    pi.code = code.data();
    pi.code_size = code.size();
    pi.entrypoint = "main0";
    pi.format = SDL_GPU_SHADERFORMAT_MSL;
    pi.num_readonly_storage_buffers = 3;
    pi.num_readwrite_storage_textures = 1;
    pi.threadcount_x = 8;
    pi.threadcount_y = 8;
    pi.threadcount_z = 1;

    state.compute_pipeline = SDL_CreateGPUComputePipeline(state.gpu_device, &pi);
    ASSERT(state.compute_pipeline);

    SDL_GPUTextureCreateInfo ti = {};
    ti.type = SDL_GPU_TEXTURETYPE_2D;
    ti.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    ti.width = state.win.bWidth;
    ti.height = state.win.bHeight;
    ti.layer_count_or_depth = 1;
    ti.num_levels = 1;
    // Add COLOR_TARGET usage so we can clear it
    ti.usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
               SDL_GPU_TEXTUREUSAGE_SAMPLER |
               SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    state.output_texture = SDL_CreateGPUTexture(state.gpu_device, &ti);
    ASSERT(state.output_texture);

    LOG("GPU initialized: texture " << state.win.bWidth << "x" << state.win.bHeight);
}

void upload_scene() {
    state.gpu_bvh_nodes.clear();
    state.gpu_triangles.clear();

    std::function<void(BVHNode*)> collect = [&](BVHNode* n) {
        if (!n) return;
        if (n->count > 0) {
            for (int i = 0; i < n->count; i++) {
                GPUTriangle gt = {};
                gt.v0[0] = n->tris[i].v0.x; gt.v0[1] = n->tris[i].v0.y; gt.v0[2] = n->tris[i].v0.z;
                gt.v1[0] = n->tris[i].v1.x; gt.v1[1] = n->tris[i].v1.y; gt.v1[2] = n->tris[i].v1.z;
                gt.v2[0] = n->tris[i].v2.x; gt.v2[1] = n->tris[i].v2.y; gt.v2[2] = n->tris[i].v2.z;
                gt.color[0] = n->mats[i].color.x; gt.color[1] = n->mats[i].color.y; gt.color[2] = n->mats[i].color.z;
                state.gpu_triangles.push_back(gt);
            }
        } else { collect(n->left); collect(n->right); }
    };
    collect(state.bvh_root);

    int idx = 0, tri_off = 0;
    flatten_bvh(state.bvh_root, state.gpu_bvh_nodes, idx, tri_off);

    LOG("Triangles: " << state.gpu_triangles.size());
    LOG("BVH Nodes: " << state.gpu_bvh_nodes.size());

    if (state.triangle_buffer) SDL_ReleaseGPUBuffer(state.gpu_device, state.triangle_buffer);
    if (state.bvh_buffer) SDL_ReleaseGPUBuffer(state.gpu_device, state.bvh_buffer);

    SDL_GPUBufferCreateInfo bi = {};
    bi.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ;
    bi.size = state.gpu_triangles.size() * sizeof(GPUTriangle);
    state.triangle_buffer = SDL_CreateGPUBuffer(state.gpu_device, &bi);
    ASSERT(state.triangle_buffer);

    bi.size = state.gpu_bvh_nodes.size() * sizeof(GPUBVHNode);
    state.bvh_buffer = SDL_CreateGPUBuffer(state.gpu_device, &bi);
    ASSERT(state.bvh_buffer);

    if (!state.camera_buffer) {
        bi.size = sizeof(CameraUniforms);
        state.camera_buffer = SDL_CreateGPUBuffer(state.gpu_device, &bi);
        ASSERT(state.camera_buffer);
    }

    SDL_GPUTransferBufferCreateInfo ti = {
        SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        static_cast<Uint32>(state.gpu_triangles.size() * sizeof(GPUTriangle))
    };
    const auto tri_t = SDL_CreateGPUTransferBuffer(state.gpu_device, &ti);
    void* td = SDL_MapGPUTransferBuffer(state.gpu_device, tri_t, false);
    memcpy(td, state.gpu_triangles.data(), ti.size);
    SDL_UnmapGPUTransferBuffer(state.gpu_device, tri_t);

    ti.size = state.gpu_bvh_nodes.size() * sizeof(GPUBVHNode);
    const auto bvh_t = SDL_CreateGPUTransferBuffer(state.gpu_device, &ti);
    void* bd = SDL_MapGPUTransferBuffer(state.gpu_device, bvh_t, false);
    memcpy(bd, state.gpu_bvh_nodes.data(), ti.size);
    SDL_UnmapGPUTransferBuffer(state.gpu_device, bvh_t);

    const auto cmd = SDL_AcquireGPUCommandBuffer(state.gpu_device);
    const auto cp = SDL_BeginGPUCopyPass(cmd);

    const SDL_GPUTransferBufferLocation tl = {tri_t, 0};
    const SDL_GPUBufferRegion tr = {
        state.triangle_buffer, 0,
        static_cast<Uint32>(state.gpu_triangles.size() * sizeof(GPUTriangle))
    };
    SDL_UploadToGPUBuffer(cp, &tl, &tr, false);

    const SDL_GPUTransferBufferLocation bl = {bvh_t, 0};
    const SDL_GPUBufferRegion br = {
        state.bvh_buffer, 0,
        static_cast<Uint32>(state.gpu_bvh_nodes.size() * sizeof(GPUBVHNode))
    };
    SDL_UploadToGPUBuffer(cp, &bl, &br, false);

    SDL_EndGPUCopyPass(cp);
    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_WaitForGPUIdle(state.gpu_device);

    SDL_ReleaseGPUTransferBuffer(state.gpu_device, tri_t);
    SDL_ReleaseGPUTransferBuffer(state.gpu_device, bvh_t);

    LOG("Scene uploaded to GPU");
}

static inline Uint32 ceilDivU32(const Uint32 a, const Uint32 b) { return (a + b - 1) / b; }

void render() {
    const float vph = 2.0f * tanf(state.cam.fov * M_PI / 360.0f);
    const float vpw = vph * static_cast<float>(state.win.bWidth) / static_cast<float>(state.win.bHeight);

    CameraUniforms cu{};
    cu.position[0] = state.cam.position.x; cu.position[1] = state.cam.position.y; cu.position[2] = state.cam.position.z;
    cu.forward[0] = state.cam.front.x; cu.forward[1] = state.cam.front.y; cu.forward[2] = state.cam.front.z;
    cu.right[0] = state.cam.right.x; cu.right[1] = state.cam.right.y; cu.right[2] = state.cam.right.z;
    cu.up[0] = state.cam.up.x; cu.up[1] = state.cam.up.y; cu.up[2] = state.cam.up.z;
    cu.vp_width = vpw;
    cu.vp_height = vph;
    cu.width = state.win.bWidth;
    cu.height = state.win.bHeight;

    SDL_GPUTransferBufferCreateInfo cam_ti{};
    cam_ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    cam_ti.size = static_cast<Uint32>(sizeof(CameraUniforms));
    SDL_GPUTransferBuffer *cam_tb = SDL_CreateGPUTransferBuffer(state.gpu_device, &cam_ti);
    ASSERT(cam_tb);

    void *cam_ptr = SDL_MapGPUTransferBuffer(state.gpu_device, cam_tb, false);
    ASSERT(cam_ptr);
    memcpy(cam_ptr, &cu, sizeof(CameraUniforms));
    SDL_UnmapGPUTransferBuffer(state.gpu_device, cam_tb);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(state.gpu_device);
    ASSERT(cmd);

    // Upload camera
    SDL_GPUCopyPass *copy_upload = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUTransferBufferLocation cam_src{ cam_tb, 0 };
    const SDL_GPUBufferRegion cam_dst{ state.camera_buffer, 0, (Uint32)sizeof(CameraUniforms) };
    SDL_UploadToGPUBuffer(copy_upload, &cam_src, &cam_dst, false);
    SDL_EndGPUCopyPass(copy_upload);

    const SDL_GPUStorageTextureReadWriteBinding rw_tex[] = {
        { state.output_texture, 0, 0, false }
    };

    SDL_GPUComputePass *comp = SDL_BeginGPUComputePass(cmd, rw_tex, 1, nullptr, 0);
    ASSERT(comp);

    SDL_BindGPUComputePipeline(comp, state.compute_pipeline);

    SDL_GPUBuffer* storage_buffers[3] = {
        state.bvh_buffer,
        state.triangle_buffer,
        state.camera_buffer
    };

    SDL_BindGPUComputeStorageBuffers(comp, 0, storage_buffers, 3);

    const Uint32 gx = ceilDivU32(static_cast<Uint32>(state.win.bWidth), 8);
    const Uint32 gy = ceilDivU32(static_cast<Uint32>(state.win.bHeight), 8);
    SDL_DispatchGPUCompute(comp, gx, gy, 1);
    SDL_EndGPUComputePass(comp);

    // Download result
    SDL_GPUTransferBufferCreateInfo dl_ti{};
    dl_ti.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    dl_ti.size = static_cast<Uint32>(state.win.bWidth * state.win.bHeight * 4);
    SDL_GPUTransferBuffer *dl_tb = SDL_CreateGPUTransferBuffer(state.gpu_device, &dl_ti);
    ASSERT(dl_tb);

    SDL_GPUCopyPass *copy_dl = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTextureRegion src_region{};
    src_region.texture = state.output_texture;
    src_region.x = 0;
    src_region.y = 0;
    src_region.z = 0;
    src_region.w = static_cast<Uint32>(state.win.bWidth);
    src_region.h = static_cast<Uint32>(state.win.bHeight);
    src_region.d = 1;

    SDL_GPUTextureTransferInfo dst_info{};
    dst_info.transfer_buffer = dl_tb;
    dst_info.offset = 0;
    dst_info.pixels_per_row = 0;  // Let SDL calculate tight packing
    dst_info.rows_per_layer = 0;  // Let SDL calculate tight packing

    SDL_DownloadFromGPUTexture(copy_dl, &src_region, &dst_info);
    SDL_EndGPUCopyPass(copy_dl);

    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_WaitForGPUIdle(state.gpu_device);

    void *dl_ptr = SDL_MapGPUTransferBuffer(state.gpu_device, dl_tb, false);
    ASSERT(dl_ptr);

    // Convert RGBA (GPU) to ARGB (SDL texture)
    const uint32_t* src = static_cast<uint32_t *>(dl_ptr);
    auto* dst = state.win.buffer;
    for (int i = 0; i < state.win.bWidth * state.win.bHeight; i++) {
        const uint32_t rgba = src[i];
        const uint8_t r = (rgba >> 0) & 0xFF;
        const uint8_t g = (rgba >> 8) & 0xFF;
        const uint8_t b = (rgba >> 16) & 0xFF;
        const uint8_t a = (rgba >> 24) & 0xFF;
        // ARGB8888 format: 0xAARRGGBB
        dst[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }

    SDL_UnmapGPUTransferBuffer(state.gpu_device, dl_tb);

    SDL_ReleaseGPUTransferBuffer(state.gpu_device, cam_tb);
    SDL_ReleaseGPUTransferBuffer(state.gpu_device, dl_tb);
}

void update() {
    if (pollEvents(&state.win, &state.input)) { state.running = false; return; }
    if (isKeyDown(&state.input, KEY_LSHIFT)) releaseMouse(state.win.window, &state.input);
    else if (!isMouseGrabbed(&state.input)) grabMouse(state.win.window, state.win.width, state.win.height, &state.input);

    int dx, dy;
    getMouseDelta(&state.input, &dx, &dy);
    cameraRotate(&state.cam, dx * state.mouse_sensitivity, -dy * state.mouse_sensitivity);

    if (isKeyDown(&state.input, KEY_W)) cameraMove(&state.cam, state.cam.front, state.move_speed);
    if (isKeyDown(&state.input, KEY_S)) cameraMove(&state.cam, mul(state.cam.front, -1), state.move_speed);
    if (isKeyDown(&state.input, KEY_A)) cameraMove(&state.cam, mul(state.cam.right, -1), state.move_speed);
    if (isKeyDown(&state.input, KEY_D)) cameraMove(&state.cam, state.cam.right, state.move_speed);
}

int main() {
    windowInit(&state.win);
    state.win.width   = WIDTH;
    state.win.height  = HEIGHT;
    state.win.bWidth  = RENDER_SCALE * WIDTH;
    state.win.bHeight = RENDER_SCALE * HEIGHT;
    state.win.title = "gpu";
    ASSERT(createWindow(&state.win));

    state.texture = SDL_CreateTexture(
        state.win.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        state.win.bWidth,
        state.win.bHeight);
    ASSERT(state.texture);

    cameraInit(&state.cam);
    state.cam.position = vec3(0, 30, 50);
    state.cam.yaw = -90;
    state.cam.pitch = -35;
    cameraUpdate(&state.cam);

    inputInit(&state.input);
    state.running = true;
    state.move_speed = 0.1f;
    state.mouse_sensitivity = 0.3f;
    state.num_models = 0;

    init_gpu();
    {
        constexpr float step = CUBE_SIZE + CUBE_PAD, half = (CUBE_GRID - 1) * step * 0.5f;
        for (int z = 0; z < CUBE_GRID; z++)
        for (int y = 0; y < CUBE_GRID; y++)
        for (int x = 0; x < CUBE_GRID; x++) {
            if (state.num_models >= MAX_MODELS) break;
            Model* c = modelCreate(state.models, &state.num_models, MAX_MODELS, vec3(x/(CUBE_GRID-1.0f), y/(CUBE_GRID-1.0f), z/(CUBE_GRID-1.0f)), 0, 0);
            ASSERT(c);
            modelLoad(c, PATH);
            modelTransform(c, vec3(x*step-half, y*step-half, z*step-half), vec3(0,0,0), vec3(CUBE_SIZE,CUBE_SIZE,CUBE_SIZE));
        }

        modelUpdate(state.models, state.num_models);
        bvh_build(&state.bvh_root, state.models, state.num_models);
    }

    upload_scene();

    while (state.running)
    {
        update();
        render();
        ASSERT(updateFramebuffer(&state.win, state.texture));

        imguiNewFrame();
            ImGui::Begin("GPU Ray Tracer");
            ImGui::Text("Pos: %.1f, %.1f, %.1f", state.cam.position.x, state.cam.position.y, state.cam.position.z);
            ImGui::Text("Yaw: %.1f, Pitch: %.1f", state.cam.yaw, state.cam.pitch);
            ImGui::Text("Front: %.2f, %.2f, %.2f", state.cam.front.x, state.cam.front.y, state.cam.front.z);
            ImGui::Text("FPS: %.1f (%.2fms)", getFPS(&state.win), getDelta(&state.win)*1000);
            ImGui::Text("Triangles: %zu", state.gpu_triangles.size());
            ImGui::Text("BVH Nodes: %zu", state.gpu_bvh_nodes.size());
            ImGui::Text("Resolution: %dx%d", state.win.bWidth, state.win.bHeight);
            ImGui::End();
        imguiEndFrame(&state.win);

        SDL_RenderPresent(state.win.renderer);
        updateFrame(&state.win);
    }

    cleanup();
    return 0;
}