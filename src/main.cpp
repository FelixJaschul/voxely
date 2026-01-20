#include <SDL3/SDL.h>
#include <imgui.h>
#include <iostream>
#include <vector>
#include <thread>
#include <future>

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

#define MAX(a, b) (( (a) > (b) ) ? (a) : (b))
#define MIN(a, b) (( (a) < (b) ) ? (a) : (b))

#define PATH "../res/cube.obj"
#define WIDTH 800
#define HEIGHT 600
#define RENDER_SCALE 0.5f
#define MAX_MODELS 64

#define CUBE_GRID   4
#define CUBE_SIZE   1.0f
#define CUBE_PAD    0.25f

// Prepared triangle
typedef struct
{
    Vec3 v0, e1, e2;
    Vec3 normal;
    Vec3 color;
} PreparedTriangle;

// State
typedef struct
{
    Window_t win;
    SDL_Texture* texture;

    Camera cam;
    Input input;

    Model models[MAX_MODELS];
    int num_models;

    std::vector<PreparedTriangle> scene_tris;
    BVHNode* bvh_root;

    bool running;
    float move_speed;
    float mouse_sensitivity;
    int num_threads;
} State;

State state = {};

#define cleanup() do { \
    if (state.bvh_root) { bvh_free(state.bvh_root); state.bvh_root = nullptr; } \
    for (int i = 0; i < state.num_models; i++) modelFree(&state.models[i]); \
    if (state.texture) SDL_DestroyTexture(state.texture); \
    destroyWindow(&state.win); \
} while(0)

#define ASSERT(x) do { \
    if(!(x)) { \
        std::cout << "Assertion failed: " << #x << " " << SDL_GetError() << std::endl; \
        state.running = false; \
        cleanup(); \
        exit(1); \
    } \
} while(0)

#define LOG(x) do { std::cout << x << std::endl; } while(0)

#define safe_inv_dir(d) \
    vec3( \
        (fabsf(d.x) > 1e-8f) ? (1.0f / d.x) : 1e30f, \
        (fabsf(d.y) > 1e-8f) ? (1.0f / d.y) : 1e30f, \
        (fabsf(d.z) > 1e-8f) ? (1.0f / d.z) : 1e30f ); \

Vec3 trace_ray(const Ray& ray)
{
    if (!state.bvh_root) return vec3(0,0,0);

    HitRecord rec = {};
    rec.hit = false;
    rec.t   = 1e30f;

    BvhRay br;
    br.origin        = ray.origin;
    br.direction     = ray.direction;
    br.inv_direction = safe_inv_dir(ray.direction);

    if (bvh_intersect(state.bvh_root, br, &rec)) return rec.mat.color;
    return vec3(0,0,0);
}

void render_frame()
{
    const float aspect = static_cast<float>(state.win.bWidth) / static_cast<float>(state.win.bHeight);
    const float vp_height = 2.0f * tanf(static_cast<float>(state.cam.fov * M_PI / 180.0f) / 2.0f);
    const float vp_width = vp_height * aspect;

    auto render_rows = [&](const int y0, const int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            for (int x = 0; x < state.win.bWidth; x++)
            {
                const float u = (static_cast<float>(x) / static_cast<float>(state.win.bWidth - 1) - 0.5f) * vp_width;
                const float v = (0.5f - static_cast<float>(y) / static_cast<float>(state.win.bHeight - 1)) * vp_height;

                const Ray ray = cameraGetRay(&state.cam, u, v);
                const auto [cx, cy, cz] = trace_ray(ray);

                const auto r = static_cast<uint8_t>(fminf(cx, 1) * 255);
                const auto g = static_cast<uint8_t>(fminf(cy, 1) * 255);
                const auto b = static_cast<uint8_t>(fminf(cz, 1) * 255);

                state.win.buffer[y * state.win.bWidth + x] = (0xFF<<24)|(r<<16)|(g<<8)|b;
            }
        }
    };

    std::vector<std::future<void>> jobs;
    const int rows = state.win.bHeight / state.num_threads;

    for (int i = 0; i < state.num_threads; i++) {
        int s = i * rows;
        int e = (i == state.num_threads-1) ? state.win.bHeight : (i+1)*rows;
        jobs.push_back(std::async(std::launch::async, render_rows, s, e));
    }
    for (auto& j : jobs) j.wait();
}

void handle_resize()
{
    if (!state.win.resized) return;

    // Update buffer dimensions based on the new window size
    state.win.bWidth  = static_cast<int>(RENDER_SCALE * static_cast<float>(state.win.width));
    state.win.bHeight = static_cast<int>(RENDER_SCALE * static_cast<float>(state.win.height));

    // Recreate framebuffer
    ASSERT(resizeBuffer(&state.win));

    // Recreate texture
    if (state.texture) SDL_DestroyTexture(state.texture);

    state.texture = SDL_CreateTexture(
        state.win.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        state.win.bWidth,
        state.win.bHeight
    );

    ASSERT(state.texture);

    LOG("Resized to " << state.win.width << "x" << state.win.height <<
        " (buffer: " << state.win.bWidth << "x" << state.win.bHeight << ")");

    state.win.resized = false;
}

void update()
{
    // Poll events (this now automatically calls updateInput)
    if (pollEvents(&state.win, &state.input)) {
        state.running = false;
        return;
    }

    // Handle window resize
    handle_resize();

    // Mouse grab control
    if (isKeyDown(&state.input, KEY_LSHIFT)) releaseMouse(state.win.window, &state.input);
    else if (!isMouseGrabbed(&state.input)) grabMouse(state.win.window, state.win.width, state.win.height, &state.input);

    // Camera rotation
    int dx, dy;
    getMouseDelta(&state.input, &dx, &dy);
    cameraRotate(&state.cam, static_cast<float>(dx) * state.mouse_sensitivity, static_cast<float>(-dy) * state.mouse_sensitivity);

    // Camera movement
    if (isKeyDown(&state.input, KEY_W)) cameraMove(&state.cam, state.cam.front, state.move_speed);
    if (isKeyDown(&state.input, KEY_S)) cameraMove(&state.cam, mul(state.cam.front,-1), state.move_speed);
    if (isKeyDown(&state.input, KEY_A)) cameraMove(&state.cam, mul(state.cam.right,-1), state.move_speed);
    if (isKeyDown(&state.input, KEY_D)) cameraMove(&state.cam, state.cam.right, state.move_speed);
}

int main() {
    windowInit(&state.win);
    state.win.width   = WIDTH;
    state.win.height  = HEIGHT;
    state.win.bWidth  = static_cast<int>(WIDTH  * RENDER_SCALE);
    state.win.bHeight = static_cast<int>(HEIGHT * RENDER_SCALE);
    state.win.title = "ray";

    ASSERT(createWindow(&state.win));

    state.texture = SDL_CreateTexture(
        state.win.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        state.win.bWidth,
        state.win.bHeight
    );

    ASSERT(state.texture);

    cameraInit(&state.cam);
    state.cam.position = vec3(0,3,10);
    state.cam.yaw = -90;
    state.cam.pitch = -20;
    cameraUpdate(&state.cam);

    inputInit(&state.input);

    state.running = true;
    state.move_speed = 0.1f;
    state.mouse_sensitivity = 0.3f;
    state.num_models = 0;
    state.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (!state.num_threads) state.num_threads = 4;

    {
        constexpr float step = CUBE_SIZE + CUBE_PAD;
        constexpr float half = (CUBE_GRID - 1) * step * 0.5f;

        for (int z = 0; z < CUBE_GRID; z++)
        {
            for (int y = 0; y < CUBE_GRID; y++)
            {
                for (int x = 0; x < CUBE_GRID; x++)
                {
                    if (state.num_models >= MAX_MODELS) break;

                    // Simple color variation for readability
                    const Vec3 color = vec3(
                        static_cast<float>(x) / (CUBE_GRID - 1),
                        static_cast<float>(y) / (CUBE_GRID - 1),
                        static_cast<float>(z) / (CUBE_GRID - 1)
                    );

                    Model* cube = modelCreate(state.models, &state.num_models, MAX_MODELS, color, 0, 0);
                    ASSERT(cube);

                    modelLoad(cube, PATH);

                    const Vec3 pos = vec3(x * step - half, y * step - half, z * step - half);

                    modelTransform(cube, pos, vec3(0, 0, 0), vec3(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE));
                }
            }
        }
    }

    {
        modelUpdate(state.models, state.num_models);
        if (state.bvh_root) bvh_free(state.bvh_root);
        bvh_build(&state.bvh_root, state.models, state.num_models);
        state.scene_tris.clear();
        for (int m = 0; m < state.num_models; m++)
        {
            const Model* model = &state.models[m];
            for (int i = 0; i < model->num_triangles; i++)
            {
                const auto& [v0, v1, v2] = model->transformed_triangles[i];
                const Vec3 e1 = sub(v1,v0);
                const Vec3 e2 = sub(v2,v0);
                state.scene_tris.push_back({ v0, e1, e2, norm(cross(e1,e2)), model->mat.color });
            }
        }
    }

    while (state.running)
    {
        update();

        render_frame();
        ASSERT(updateFramebuffer(&state.win, state.texture));

        imguiNewFrame();
            ImGui::Begin("status");
            ImGui::Text("Path: %s", PATH);
            ImGui::Text("Camera pos: %.2f, %.2f, %.2f", state.cam.position.x, state.cam.position.y, state.cam.position.z);
            ImGui::Text("Fps: %.2f", getFPS(&state.win));
            ImGui::Text("Delta: %.4f ms", getDelta(&state.win) * 1000.0);
            ImGui::Text("Triangles: %zu", state.scene_tris.size());
            ImGui::Text("Resolution: %dx%d (buffer: %dx%d)", state.win.width, state.win.height, state.win.bWidth, state.win.bHeight);
            ImGui::End();
        imguiEndFrame(&state.win);

        SDL_RenderPresent(state.win.renderer);
        updateFrame(&state.win);
    }

    // Cleanup
    cleanup();
    return 0;
}