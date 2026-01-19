#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <iostream>
#include <vector>
#include <thread>
#include <future>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CORE_IMPLEMENTATION
#define MATH_IMPLEMENTATION
#define KEYS_IMPLEMENTATION
#define CAMERA_IMPLEMENTATION
#define MODEL_IMPLEMENTATION
#define SDL_IMPLEMENTATION
#include <wrapper/core.h>

typedef struct {
    Vec3 v0, e1, e2, normal, color;
} PreparedTriangle;

bool ray_prepared_triangle_intersect(const Ray &ray, const PreparedTriangle* tri, float* t) {
    const Vec3 h = cross(ray.direction, tri->e2);
    const float a = dot(tri->e1, h);
    if (a > -0.00001f && a < 0.00001f)
        return false;
    const float f = 1.0f / a;
    const Vec3 s = sub(ray.origin, tri->v0);
    const float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;
    const Vec3 q = cross(s, tri->e1);
    if (const float v = f * dot(ray.direction, q);
        v < 0.0f || u + v > 1.0f) return false;
    if (const float _t = f * dot(tri->e2, q);
        _t > 0.00001f) {
        *t = _t;
        return true;
    }
    return false;
}

#define ASSERT(x) do { if(!(x)) std::cout << "Assertion failed: " << #x << std::endl; } while(0)
#define LOG(x) do { std::cout << x << std::endl; } while(0)

#define MAX(a, b) (( (a) > (b) ) ? (a) : (b))
#define MIN(a, b) (( (a) < (b) ) ? (a) : (b))

#define WIDTH 298*2
#define HEIGHT 198*2
#define RENDER_SCALE 0.5f

#define CHUNK_SIZE 32
#define CHUNK_VOL (CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE)

float mouse_sensitivity = 0.002f;

typedef struct { Vec3 min, max; } AABB;

typedef struct
{
    int width;
    int height;
    uint32_t* pixels;
} fb_t;

#define MAX_MODELS 16

typedef struct 
{
    Window_t win;
    int state;
    fb_t fb;
    Camera cam;
    Input input;
    Model models[MAX_MODELS];
    int num_models;
    SDL_Texture* texture;
} state_t;

state_t state;

#define present_frame() do { \
    void* _pixels; \
    int _pitch; \
    SDL_LockTexture(state.texture, nullptr, &_pixels, &_pitch); \
    memcpy(_pixels, state.fb.pixels, state.fb.width * state.fb.height * 4); \
    SDL_UnlockTexture(state.texture); \
    SDL_RenderTexture(state.win.renderer, state.texture, nullptr, nullptr); \
} while(0)

int main()
{
    windowInit(&state.win);
    state.win.width = WIDTH * 2;
    state.win.height = HEIGHT * 2;
    state.win.title = "renderer";
    if (!createWindow(&state.win)) return 1;

    state.fb.width  = static_cast<int>(WIDTH * RENDER_SCALE);
    state.fb.height = static_cast<int>(HEIGHT * RENDER_SCALE);
    state.fb.pixels = static_cast<uint32_t *>(malloc(state.fb.width * state.fb.height * 4));

    state.texture = SDL_CreateTexture(state.win.renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, state.fb.width, state.fb.height);
    ASSERT(state.texture);

    Camera camera;
    cameraInit(&camera);
    camera.position = vec3(0.0f, 0.0f, 2.0f);
    camera.yaw = -90.0f;

    Input input;
    inputInit(&input);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForSDLRenderer(state.win.window, state.win.renderer);
    ImGui_ImplSDLRenderer3_Init(state.win.renderer);

    cameraInit(&state.cam);
    state.cam.position = vec3(0, 0, 5);
    state.cam.yaw = -90.0f; // Look towards -Z
    cameraUpdate(&state.cam);

    state.num_models = 0;
    if (Model* cube = modelCreate(state.models, &state.num_models, MAX_MODELS, vec3(1,0,0), 0.5f, 1.0f)) {
        modelLoad(cube, "../res/cube.obj");
        modelTransform(cube, vec3(0,0,0), vec3(0.4f, 0.4f, 0), vec3(2,2,2));
    }

    state.state = 0;
    while (state.state == 0)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) state.state = 1;

            const bool shift_down = (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
            if (SDL_GetWindowRelativeMouseMode(state.win.window) != shift_down) {
                SDL_SetWindowRelativeMouseMode(state.win.window, shift_down);
            }

            if (event.type == SDL_EVENT_MOUSE_MOTION && shift_down) {
                cameraRotate(&state.cam, event.motion.xrel * 0.5f, -event.motion.yrel * 0.5f);
            }
        }

        const float move_speed = 0.03f;
        if (isKeyDown(&input, KEY_W)) cameraMove(&camera, camera.front, move_speed);
        if (isKeyDown(&input, KEY_S)) cameraMove(&camera, mul(camera.front, -1), move_speed);
        if (isKeyDown(&input, KEY_A)) cameraMove(&camera, mul(camera.right, -1), move_speed);
        if (isKeyDown(&input, KEY_D)) cameraMove(&camera, camera.right, move_speed);

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("raytrace");
        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::Text("%dx%d, %dx%d (Scale: %.2f)", state.win.width, state.win.height, state.fb.width, state.fb.height, RENDER_SCALE);
        if (ImGui::Button("Quit")) state.state = 1;
        ImGui::End();

        // Render models
        modelUpdate(state.models, state.num_models);

        Vec3 light_dir = norm(vec3(1, 1, 1));
        std::vector<PreparedTriangle> prepared_tris;
        for (int m = 0; m < state.num_models; m++)
        {
            const Model* model = &state.models[m];
            for (int i = 0; i < model->num_triangles; i++)
            {
                auto [v0, v1, v2] = model->transformed_triangles[i];
                const Vec3 e1 = sub(v1, v0);
                const Vec3 e2 = sub(v2, v0);
                const Vec3 n = norm(cross(e1, e2));
                prepared_tris.push_back({v0, e1, e2, n, model->mat.color});
            }
        }

        float aspect_ratio = (float)state.fb.width / (float)state.fb.height;
        float viewport_height = 2.0f * tanf((state.cam.fov * (float)M_PI / 180.0f) / 2.0f);
        float viewport_width = viewport_height * aspect_ratio;

        auto render_rows = [&](const int start_y, const int end_y) {
            for (int y = start_y; y < end_y; y++) {
                for (int x = 0; x < state.fb.width; x++) {
                    const float u_scaled = (static_cast<float>(x) / static_cast<float>(state.fb.width - 1) - 0.5f) * viewport_width;
                    const float v_scaled = (0.5f - static_cast<float>(y) / static_cast<float>(state.fb.height - 1)) * viewport_height;

                    Ray ray = cameraGetRay(&state.cam, u_scaled, v_scaled);

                    float min_t = 1e10f;
                    Vec3 hit_color = vec3(0.2f, 0.4f, 0.6f); // Background color

                    for (const auto& tri : prepared_tris)
                    {
                        float t;
                        if (ray_prepared_triangle_intersect(ray, &tri, &t))
                        {
                            if (t < min_t)
                            {
                                min_t = t;
                                // Simple diffuse shading
                                float diff = MAX(0.2f, dot(tri.normal, light_dir));
                                hit_color = mul(tri.color, diff);
                            }
                        }
                    }

                    const uint8_t r = static_cast<uint8_t>((MIN(hit_color.x, 1.0f) * 255));
                    const uint8_t g = static_cast<uint8_t>((MIN(hit_color.y, 1.0f) * 255));
                    const uint8_t b = static_cast<uint8_t>((MIN(hit_color.z, 1.0f) * 255));
                    state.fb.pixels[y * state.fb.width + x] = (0xFF << 24) | (r << 16) | (g << 8) | b;
                }
            }
        };

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < state.fb.height; y++) {
            render_rows(y, y + 1);
        }
#else
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
        std::vector<std::future<void>> futures;
        int rows_per_thread = state.fb.height / num_threads;
        for (int i = 0; i < num_threads; i++)
        {
            int start_y = i * rows_per_thread;
            int end_y = (i == num_threads - 1) ? state.fb.height : (i + 1) * rows_per_thread;
            futures.push_back(std::async(std::launch::async, render_rows, start_y, end_y));
        }
        for (auto& f : futures) f.wait();
#endif

        ImGui::Render();
        SDL_SetRenderDrawColor(state.win.renderer, 0, 0, 0, 255);
        SDL_RenderClear(state.win.renderer);

        present_frame();

        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), state.win.renderer);
        SDL_RenderPresent(state.win.renderer);
    }

    free(state.fb.pixels);
    for (auto &model : state.models) modelFree(&model);
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyTexture(state.texture);
    SDL_DestroyRenderer(state.win.renderer);
    SDL_DestroyWindow(state.win.window);
    SDL_Quit();

    return 0;
}
