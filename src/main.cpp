#include <SDL3/SDL.h>
#include <imgui.h>
#include <iostream>
#define CORE_IMPLEMENTATION
#define MATH_IMPLEMENTATION
#define KEYS_IMPLEMENTATION
#define CAMERA_IMPLEMENTATION
#define MODEL_IMPLEMENTATION
#define RENDER3D_IMPLEMENTATION
#define SDL_IMPLEMENTATION
#define IMGUI_IMPLEMENTATION
#include "wrapper/core.h"
#define UTIL_IMPLEMENTATION // BVH
#include "bvh.h"

#define CUBE_GRID   40
#define CUBE_SIZE   1.0f
#define CUBE_PAD    0.25f
#define PATH "../res/cube.obj"
#define WIDTH 1250
#define HEIGHT 850
#define RENDER_SCALE 0.5f
#define MAX_MODELS CUBE_GRID*CUBE_GRID*CUBE_GRID +100

#define GRID_SIZE CUBE_GRID
uint8_t voxel_grid[GRID_SIZE][GRID_SIZE][GRID_SIZE] = {0};

static HitRecord grid_traverse(const BvhRay &ray)
{
    constexpr float step = CUBE_SIZE + CUBE_PAD;
    constexpr float gridSizeWorld = (GRID_SIZE) * step;
    constexpr float half = gridSizeWorld * 0.5f;

    Vec3 gridMin = vec3(-half, -half, -half);

    const auto [rtgx, rtgy, rtgz] = sub(ray.origin, gridMin);

    const int exitX = ray.direction.x > 0 ? GRID_SIZE : -1;
    const int exitY = ray.direction.y > 0 ? GRID_SIZE : -1;
    const int exitZ = ray.direction.z > 0 ? GRID_SIZE : -1;

    int x = static_cast<int>(floorf(rtgx / step));
    int y = static_cast<int>(floorf(rtgy / step));
    int z = static_cast<int>(floorf(rtgz / step));

    const float tx = (ray.direction.x > 0) ? (x + 1) * step : x * step;
    const float ty = (ray.direction.y > 0) ? (y + 1) * step : y * step;
    const float tz = (ray.direction.z > 0) ? (z + 1) * step : z * step;

    float tMaxX = (tx - rtgx) / ray.direction.x;
    float tMaxY = (ty - rtgy) / ray.direction.y;
    float tMaxZ = (tz - rtgz) / ray.direction.z;

    const float tDeltaX = step / fabsf(ray.direction.x);
    const float tDeltaY = step / fabsf(ray.direction.y);
    const float tDeltaZ = step / fabsf(ray.direction.z);

    const int stepX = (ray.direction.x > 0) ? 1 : -1;
    const int stepY = (ray.direction.y > 0) ? 1 : -1;
    const int stepZ = (ray.direction.z > 0) ? 1 : -1;
    
    HitRecord hit_rec = { .hit = false, .t = FLT_MAX };

    while (true) 
    {
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE && z >= 0 && z < GRID_SIZE) 
        {
            if (voxel_grid[z][y][x]) 
            {
                hit_rec.hit = true;
                
                if (tMaxX < tMaxY) {
                    if (tMaxX < tMaxZ) hit_rec.t = tMaxX;
                    else hit_rec.t = tMaxZ;
                } else {
                    if (tMaxY < tMaxZ) hit_rec.t = tMaxY;
                    else hit_rec.t = tMaxZ;
                }

                const float fx = (x - GRID_SIZE/2.f);
                const float fy = (y - GRID_SIZE/2.f);
                const float fz = (z - GRID_SIZE/2.f);
                hit_rec.normal = norm(vec3(fx, fy, fz));
                
                hit_rec.color = vec3(x / static_cast<float>(GRID_SIZE), y / static_cast<float>(GRID_SIZE), z / static_cast<float>(GRID_SIZE));

                return hit_rec;
            }
        }

        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
                if (x == exitX) return hit_rec;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
                if (z == exitZ) return hit_rec;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
                if (y == exitY) return hit_rec;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
                if (z == exitZ) return hit_rec;
            }
        }
    }
}


typedef struct {
    Window_t win;
    Renderer3D renderer;
    SDL_Texture *texture;
    Camera cam;
    Input input;
    Model models[MAX_MODELS];
    int num_models;
    BVHNode* bvh_root;
    bool running;
    bool faster;
} State;

State state = {};

#define ASSERT(x) do { if(!(x)) { std::cout << "Error: " << #x << " " << SDL_GetError() << std::endl; cleanup(); exit(1); } } while(0)
#define LOG(x) do { std::cout << x << std::endl; } while(0)

#define cleanup() do { \
    for (int i = 0; i < state.num_models; i++) modelFree(&state.models[i]); \
    render3DFree(&state.renderer); \
    if (state.texture) SDL_DestroyTexture(state.texture); \
    destroyWindow(&state.win); \
} while(0)

void update()
{
    if (pollEvents(&state.win, &state.input))
    {
        state.running = false;
        return;
    }

    if (isKeyDown(&state.input, KEY_SPACE)) releaseMouse(state.win.window, &state.input);
    else if (!isMouseGrabbed(&state.input)) grabMouse(state.win.window, state.win.width, state.win.height, &state.input);
    if (isKeyDown(&state.input, KEY_LSHIFT)) state.faster = true;
    else state.faster = false;

    float speed;
    if (state.faster) speed = 3.0f;
    else speed = 0.1f;
    constexpr float sensi = 0.3f;
    int dx, dy;
    getMouseDelta(&state.input, &dx, &dy);
    cameraRotate(&state.cam, dx * sensi, -dy * sensi);

    if (isKeyDown(&state.input, KEY_W)) cameraMove(&state.cam, state.cam.front, speed);
    if (isKeyDown(&state.input, KEY_S)) cameraMove(&state.cam, mul(state.cam.front, -1), speed);
    if (isKeyDown(&state.input, KEY_A)) cameraMove(&state.cam, mul(state.cam.right, -1), speed);
    if (isKeyDown(&state.input, KEY_D)) cameraMove(&state.cam, state.cam.right, speed);
}

void render()
{
    memset(state.win.buffer, 0, state.win.bWidth * state.win.bHeight * sizeof(uint32_t));
    
    // Ray casting
    #pragma omp parallel for
    for (int j = 0; j < state.win.bHeight; j++)
    {
        for (int i = 0; i < state.win.bWidth; i++)
        {
            float u = (2.0f * i - state.win.bWidth) / state.win.bHeight;
            float v = -((2.0f * j - state.win.bHeight) / state.win.bHeight);

            Vec3 dir = norm(add(state.cam.front, add(mul(state.cam.right, u), mul(state.cam.up, v))));

            auto safe_inv = [](float d) {
                const float eps = 1e-6f;
                if (fabsf(d) < eps) return (d >= 0.0f ? 1.0f : -1.0f) * 1e30f; // big number
                return 1.0f / d;
            };

            BvhRay ray = {
                .origin = state.cam.position,
                .direction = dir,
                .inv_direction = vec3(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z))
            };

            // Voxel grid intersection
            HitRecord voxel_hit = grid_traverse(ray);

            HitRecord rec = { .hit = false, .t = FLT_MAX };
            if (state.bvh_root) {
                bvh_intersect(state.bvh_root, ray, &rec);
            }

            uint32_t color = 0;
            if (voxel_hit.hit && voxel_hit.t < rec.t) {
                 float diff = fmaxf(0.0f, dot(voxel_hit.normal, state.renderer.light_dir));
                 Vec3 c = mul(voxel_hit.color, diff);
                 color = 0xFF000000 | ((int)(c.x * 255) << 16) | ((int)(c.y * 255) << 8) | (int)(c.z * 255);
            } else if (rec.hit) {
                 float diff = fmaxf(0.0f, dot(rec.normal, state.renderer.light_dir));
                 Vec3 c = mul(rec.mat.color, diff);
                 color = 0xFF000000 | ((int)(c.x * 255) << 16) | ((int)(c.y * 255) << 8) | (int)(c.z * 255);
            }
            
            state.win.buffer[j * state.win.bWidth + i] = color;
        }
    }


    ASSERT(updateFramebuffer(&state.win, state.texture));

    imguiNewFrame();
        ImGui::Begin("3D Rasterizer");
        ImGui::Text("Pos: %.1f, %.1f, %.1f", state.cam.position.x, state.cam.position.y, state.cam.position.z);
        ImGui::Text("Yaw: %.1f, Pitch: %.1f", state.cam.yaw, state.cam.pitch);
        ImGui::Text("Front: %.2f, %.2f, %.2f", state.cam.front.x, state.cam.front.y, state.cam.front.z);
        ImGui::Text("FPS: %.1f (%.2fms)", getFPS(&state.win), getDelta(&state.win) * 1000);
        ImGui::Text("Models: %d", state.num_models);
        ImGui::Text("Resolution: %dx%d", state.win.bWidth, state.win.bHeight);
        ImGui::Separator();
        ImGui::End();
    imguiEndFrame(&state.win);

    SDL_RenderPresent(state.win.renderer);
    updateFrame(&state.win);
}

int main()
{
    windowInit(&state.win);
    state.win.width   = WIDTH;
    state.win.height  = HEIGHT;
    state.win.bWidth  = RENDER_SCALE * WIDTH;
    state.win.bHeight = RENDER_SCALE * HEIGHT;
    state.win.title = "GPU";
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

    render3DInit(&state.renderer, &state.win, &state.cam);
    state.renderer.light_dir = vec3(0.3f, -1.0f, 0.5f);

    state.running = true;
    state.num_models = 0;
    state.bvh_root = NULL;

    {
        LOG("Building " << GRID_SIZE << "x" << GRID_SIZE << "x" << GRID_SIZE << " voxel grid...");
        for (int z = 0; z < GRID_SIZE; z++)
        for (int y = 0; y < GRID_SIZE; y++)
        for (int x = 0; x < GRID_SIZE; x++)
        {
            // Simple sphere for now
            float fx = (x - GRID_SIZE/2.f);
            float fy = (y - GRID_SIZE/2.f);
            float fz = (z - GRID_SIZE/2.f);
            if (sqrtf(fx*fx + fy*fy + fz*fz) < GRID_SIZE/2.f) voxel_grid[z][y][x] = 1;
        }
        LOG("Created voxel grid");
    }

    Model *lightCube = modelCreate(state.models, &state.num_models, MAX_MODELS, vec3(1, 1, 1), 0.0f, 0.0f);
    ASSERT(lightCube);
    modelLoad(lightCube, PATH);

    bvh_build(&state.bvh_root, state.models, state.num_models);

    while (state.running)
    {
        update();

        {
            static float lightAngle = 0.0f;
            lightAngle += static_cast<float>(getDelta(&state.win)) * 0.2f;
            state.renderer.light_dir = norm(vec3(-cosf(lightAngle), -0.35f, -sinf(lightAngle)));
            constexpr float radius = 120.0f;
            const Vec3 lightPos = vec3(cosf(lightAngle) * radius, 60.0f, sinf(lightAngle) * radius);
            constexpr float lightSize = 8.0f;
            modelTransform(lightCube, lightPos, vec3(0, 0, 0), vec3(lightSize, lightSize, lightSize));
        }

        modelUpdate(lightCube, 1);
        render();
    }

    bvh_free(state.bvh_root);
    cleanup();
    return 0;
}
