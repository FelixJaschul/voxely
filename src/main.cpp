#include <SDL3/SDL.h>
#include <imgui.h>
#include <iostream>

#define CORE_IMPLEMENTATION
#define MATH_IMPLEMENTATION
#define KEYS_IMPLEMENTATION
#define CAMERA_IMPLEMENTATION
#define SDL_IMPLEMENTATION
#define IMGUI_IMPLEMENTATION
#define RENDER3D_IMPLEMENTATION
#include "wrapper/core.h"

#define GRID_SIZE 500
#define WIDTH 1250
#define HEIGHT 850

// VOXEL DATA STRUCTURE
struct VoxelGrid {
    uint8_t data[GRID_SIZE][GRID_SIZE][GRID_SIZE];
    int size;

    void init()
    {
        size = GRID_SIZE;
        memset(data, 0, sizeof(data));
    }

    void setSphere(const float radius)
    {
        for (int z = 0; z < size; z++)
        for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
            data[z][y][x] = ((x - size * 0.5f)*(x - size * 0.5f) + (y - size * 0.5f)*(y - size * 0.5f) + (z - size * 0.5f)*(z - size * 0.5f) < radius*radius) ? 1 : 0;
    }

    [[nodiscard]] bool at(const int x, const int y, const int z) const
    {
        if (x < 0 || y < 0 || z < 0) return false;
        if (x >= size || y >= size || z >= size) return false;
        return data[z][y][x] != 0;
    }
};

static void buildVoxelModel(Model* m, const VoxelGrid* g)
{
    const int maxVox = g->size * g->size * g->size;
    const int maxTris = maxVox * 12;

    free(m->transformed_triangles);
    m->transformed_triangles = static_cast<Triangle *>(malloc(sizeof(Triangle) * maxTris));
    m->num_triangles = 0;
    m->mat.color = vec3(0.7f, 0.8f, 1.0f);

    auto V = [&](const float x, const float y, const float z) { return vec3(x - g->size * 0.5f, y - g->size * 0.5f, z - g->size * 0.5f); };
    auto T = [&](Triangle* out, int& n, const Vec3 a, const Vec3 b, const Vec3 c) { out[n++] = {a, b, c}; };

    const int offsets[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
    const int faces[6][6] = { {0,4,6, 0,6,2}, {1,3,7, 1,7,5}, {0,1,5, 0,5,4}, {2,6,7, 2,7,3}, {0,2,3, 0,3,1}, {4,5,7, 4,7,6} };

    for (int z=0; z<g->size; z++)
    for (int y=0; y<g->size; y++)
    for (int x=0; x<g->size; x++)
    {
        if (!g->at(x,y,z)) continue;
        Vec3 P[8] = { V(x, y, z), V(x+1, y, z), V(x, y+1, z), V(x+1, y+1, z), V(x, y, z+1), V(x+1, y, z+1), V(x, y+1, z+1), V(x+1, y+1, z+1) };

        for (int f=0; f<6; f++)
        {
            const int ny = y+offsets[f][1];
            const int nz = z+offsets[f][2];
            const int nx = x+offsets[f][0];
            if (!g->at(nx, ny, nz))
            {
                T(m->transformed_triangles, m->num_triangles, P[faces[f][0]], P[faces[f][1]], P[faces[f][2]]);
                T(m->transformed_triangles, m->num_triangles, P[faces[f][3]], P[faces[f][4]], P[faces[f][5]]);
            }
        }
    }

    if (m->num_triangles > 0)
        m->transformed_triangles = static_cast<Triangle *>(realloc(m->transformed_triangles, sizeof(Triangle) * m->num_triangles));
}

struct State {
    Window_t win;
    Renderer3D r;
    SDL_Texture* texture;
    Camera cam;
    Input input;
    VoxelGrid voxels;
    Model voxelModel;
    bool running;
    bool faster;
};

static State state = {};

int main()
{
    memset(&state, 0, sizeof(state));

    windowInit(&state.win);
    state.win.width = WIDTH;
    state.win.height = HEIGHT;
    state.win.bWidth = WIDTH;
    state.win.bHeight = HEIGHT;
    state.win.title = "voxely";
    ASSERT(createWindow(&state.win));

    // Streaming texture used for uploading the CPU framebuffer each frame
    state.texture = SDL_CreateTexture(
        state.win.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        state.win.bWidth,
        state.win.bHeight
    );
    ASSERT(state.texture);

    cameraInit(&state.cam);
    state.cam.position = vec3(0, 30, 400);
    state.cam.yaw = -90;
    state.cam.pitch = -20;
    cameraUpdate(&state.cam);

    inputInit(&state.input);

    state.voxels.init();
    state.voxels.setSphere(GRID_SIZE * 0.4f);

    memset(&state.voxelModel, 0, sizeof(state.voxelModel));
    buildVoxelModel(&state.voxelModel, &state.voxels);

    render3DInit(&state.r, &state.win, &state.cam);

    state.r.light_dir = vec3(0.3f, -1.0f, 0.5f);
    state.running = true;

    while (state.running) {
        {
            if (pollEvents(&state.win, &state.input)) {
                state.running = false; break;
            }

            if (isKeyDown(&state.input, KEY_SPACE)) releaseMouse(state.win.window, &state.input);
            else if (!isMouseGrabbed(&state.input)) grabMouse(state.win.window, state.win.width, state.win.height, &state.input);

            state.faster = isKeyDown(&state.input, KEY_LSHIFT);
            const float speed = state.faster ? 2.0f : 0.1f;

            int dx, dy;
            getMouseDelta(&state.input, &dx, &dy);
            cameraRotate(&state.cam, dx * 0.3f, -dy * 0.3f);

            if (isKeyDown(&state.input, KEY_W)) cameraMove(&state.cam, state.cam.front, speed);
            if (isKeyDown(&state.input, KEY_S)) cameraMove(&state.cam, mul(state.cam.front, -1), speed);
            if (isKeyDown(&state.input, KEY_A)) cameraMove(&state.cam, mul(state.cam.right, -1), speed);
            if (isKeyDown(&state.input, KEY_D)) cameraMove(&state.cam, state.cam.right, speed);
        }
        {
            // Rotate light
            static float lightAngle = 0.0f;
            lightAngle += getDelta(&state.win) * 0.2f;
            state.r.light_dir = norm(vec3(-cosf(lightAngle), -0.35f, -sinf(lightAngle)));
            render3DClear(&state.r);
            render3DModel(&state.r, &state.voxelModel);

            ASSERT(updateFramebuffer(&state.win, state.texture));

            imguiNewFrame();
            ImGui::Begin("voxely");
            ImGui::Text("Pos: %.1f, %.1f, %.1f", state.cam.position.x, state.cam.position.y, state.cam.position.z);
            ImGui::Text("FPS: %.1f (%.2fms)", getFPS(&state.win), getDelta(&state.win) * 1000);
            ImGui::Text("Grid: %dx%dx%d", GRID_SIZE, GRID_SIZE, GRID_SIZE);
            ImGui::Text("Tris: %d", state.voxelModel.num_triangles);
            ImGui::End();
            imguiEndFrame(&state.win);

            SDL_RenderPresent(state.win.renderer);
            updateFrame(&state.win);
        }
    }

    // Cleanup
    render3DFree(&state.r);

    if (state.voxelModel.transformed_triangles) {
        free(state.voxelModel.transformed_triangles);
        state.voxelModel.transformed_triangles = nullptr;
        state.voxelModel.num_triangles = 0;
    }

    SDL_DestroyTexture(state.texture);
    destroyWindow(&state.win);
    return 0;
}
