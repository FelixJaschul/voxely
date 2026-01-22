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

#define CUBE_GRID   50
#define CUBE_SIZE   1.0f
#define CUBE_PAD    0.25f
#define PATH "../res/cube.obj"
#define WIDTH 1250
#define HEIGHT 850
#define RENDER_SCALE 0.5f
#define MAX_MODELS CUBE_GRID*CUBE_GRID*CUBE_GRID +1

typedef struct {
    Window_t win;
    Renderer3D renderer;
    SDL_Texture *texture;
    Camera cam;
    Input input;
    Model models[MAX_MODELS];
    int num_models;
    bool cam_to_light;
    bool running;
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

    if (isKeyDown(&state.input, KEY_LSHIFT)) releaseMouse(state.win.window, &state.input);
    else if (!isMouseGrabbed(&state.input)) grabMouse(state.win.window, state.win.width, state.win.height, &state.input);

    constexpr float speed = 0.1f;
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
    render3DClear(&state.renderer);

    render3DScene(&state.renderer, state.models, state.num_models);

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
        ImGui::Checkbox("Wireframe", &state.renderer.wireframe);
        ImGui::Checkbox("Backface Culling", &state.renderer.backface_culling);
        ImGui::Checkbox("Camera To Light", &state.cam_to_light);
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
    state.renderer.light_dir = vec3(0.0f, 0.0f, 0.0f);

    state.running = true;
    state.num_models = 0;

    {
        constexpr float step = CUBE_SIZE + CUBE_PAD;
        constexpr float half = (CUBE_GRID - 1) * step * 0.5f;

        LOG("Building " << CUBE_GRID << "x" << CUBE_GRID << "x" << CUBE_GRID << " cube grid...");

        for (int z = 0; z < CUBE_GRID; z++)
            for (int y = 0; y < CUBE_GRID; y++)
                for (int x = 0; x < CUBE_GRID; x++)
                {
                    if (state.num_models >= MAX_MODELS) break;

                    Model *c = modelCreate(
                        state.models, &state.num_models, MAX_MODELS,
                        vec3(x / (CUBE_GRID - 1.0f), y / (CUBE_GRID - 1.0f), z / (CUBE_GRID - 1.0f)),
                        0, 0);
                    ASSERT(c);

                    modelLoad(c, PATH);
                    modelTransform(c,
                        vec3(x * step - half, y * step - half, z * step - half), vec3(0, 0, 0),
                        vec3(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE));
                }

        modelUpdate(state.models, state.num_models);
        LOG("Created " << state.num_models << " models");
    }

    Model *lightCube = modelCreate(state.models, &state.num_models, MAX_MODELS, vec3(1, 1, 1), 0.0f, 0.0f);
    ASSERT(lightCube);
    modelLoad(lightCube, PATH);

    while (state.running)
    {
        update();
        static float lightAngle = 0.0f;
        lightAngle += static_cast<float>(getDelta(&state.win)) * 0.2f;

        state.renderer.light_dir = norm(vec3(cosf(lightAngle), -0.35f, sinf(lightAngle)));

        constexpr float radius = 120.0f;
        const Vec3 lightPos = vec3(cosf(lightAngle) * radius, 60.0f, sinf(lightAngle) * radius);

        constexpr float lightSize = 8.0f;
        modelTransform(lightCube, lightPos, vec3(0, 0, 0), vec3(lightSize, lightSize, lightSize));

        modelUpdate(lightCube, 1);
        if (state.cam_to_light) state.cam.position = lightPos;
        render();
    }

    cleanup();
    return 0;
}
