#include "../lib/SDL/include/SDL3/SDL.h"
#include "../lib/imgui/imgui.h"
#include <iostream>

#define CORE_IMPLEMENTATION
#define MATH_IMPLEMENTATION
#define KEYS_IMPLEMENTATION
#define CAMERA_IMPLEMENTATION
#define SDL_IMPLEMENTATION
#define IMGUI_IMPLEMENTATION
#define RENDER3D_IMPLEMENTATION
#include "../lib/wrapper/core.h"

#define GRID_SIZE 200
#define WIDTH 2100
#define HEIGHT 1300

// VOXEL DATA STRUCTURE
struct VoxelGrid
{
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

    void setCube(const int cx, const int cy, const int cz, int size)
    {
        const int half = size / 2;
        for (int z = cz - half; z <= cz + half; z++)
        for (int y = cy - half; y <= cy + half; y++)
        for (int x = cx - half; x <= cx + half; x++)
            if (x >= 0 && x < this->size &&
                y >= 0 && y < this->size &&
                z >= 0 && z < this->size)
                data[z][y][x] = 1;
    }

    // Helper functions
    static float clamp(const float x, const float min, const float max) {
        return fmaxf(min, fminf(max, x));
    }

    static float mix(const float a, const float b, const float t)
    {
        return a + t * (b - a);
    }

    static float smoothstep(const float edge0, const float edge1, float x)
    {
        x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
        return x * x * (3 - 2 * x);
    }

    static float fract(const float x)
    {
        return x - floorf(x);
    }

    static float hash3(const float x, const float y, const float z)
    {
        return fract(sin(dot(vec3(x, y, z), vec3(12.9898, 78.233, 45.164))) * 43758.5453);
    }

    float noise3(const float x, const float y, const float z)
    {
        const int ix = static_cast<int>(floorf(x));
        const int iy = static_cast<int>(floorf(y));
        const int iz = static_cast<int>(floorf(z));

        const float fx = x - ix;
        const float fy = y - iy;
        const float fz = z - iz;

        const float n000 = hash3(ix, iy, iz);
        const float n100 = hash3(ix + 1, iy, iz);
        const float n010 = hash3(ix, iy + 1, iz);
        const float n110 = hash3(ix + 1, iy + 1, iz);
        const float n001 = hash3(ix, iy, iz + 1);
        const float n101 = hash3(ix + 1, iy, iz + 1);
        const float n011 = hash3(ix, iy + 1, iz + 1);
        const float n111 = hash3(ix + 1, iy + 1, iz + 1);

        const float u = smoothstep(0.0f, 1.0f, fx);
        const float v = smoothstep(0.0f, 1.0f, fy);
        const float w = smoothstep(0.0f, 1.0f, fz);

        const float nx00 = mix(n000, n100, u);
        const float nx10 = mix(n010, n110, u);
        const float nx01 = mix(n001, n101, u);
        const float nx11 = mix(n011, n111, u);

        const float ny0 = mix(nx00, nx10, v);
        const float ny1 = mix(nx01, nx11, v);

        return mix(ny0, ny1, w);
    }

    void setRandomNoiseSponge()
    {
        memset(data, 0, sizeof(data));
        constexpr float scale = 10.0f;

        for (int z = 0; z < size; z++)
        for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            const float nx = static_cast<float>(x) / size;
            const float ny = static_cast<float>(y) / size;
            const float nz = static_cast<float>(z) / size;
            if (noise3(nx * scale, ny * scale, nz * scale) > 0.4f) data[z][y][x] = 1;
        }
    }

    [[nodiscard]] bool at(const int x, const int y, const int z) const
    {
        if (x < 0 || y < 0 || z < 0) return false;
        if (x >= size || y >= size || z >= size) return false;
        return data[z][y][x] != 0;
    }
};

struct State {
    Window_t win;
    Renderer r;
    SDL_Texture* texture;
    Camera cam;
    Input input;
    VoxelGrid voxels;
    Model voxelModel;
    bool running;
    bool faster;
    bool light_rot;
};

static State state = {};

static void buildVoxelModel(Model* m, const VoxelGrid* g)
{
    auto V = [&](const float x, const float y, const float z) {
        return vec3(x - g->size * 0.5f, y - g->size * 0.5f, z - g->size * 0.5f);
    };

    int tri_count = 0;
    const int offsets[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };

    for (int z = 0; z < g->size; z++)
    for (int y = 0; y < g->size; y++)
    for (int x = 0; x < g->size; x++) {
        if (!g->at(x, y, z)) continue;
        for (const auto offset : offsets) {
            const int nx = x + offset[0];
            const int ny = y + offset[1];
            const int nz = z + offset[2];
            if (!g->at(nx, ny, nz)) tri_count += 2; // 2 triangles per face
        }
    }

    // Free old data
    if (m->transformed_triangles) {
        free(m->transformed_triangles);
        m->transformed_triangles = nullptr;
    }

    // Early exit if no geometry
    if (tri_count == 0) {
        m->num_triangles = 0;
        return;
    }

    m->transformed_triangles = static_cast<Triangle*>(malloc(sizeof(Triangle) * tri_count));
    m->num_triangles = 0;

    // Face vertex indices (matches your P[8] layout)
    const int faces[6][6] = {
        {0,4,6, 0,6,2}, // -X
        {1,3,7, 1,7,5}, // +X
        {0,1,5, 0,5,4}, // -Y
        {2,6,7, 2,7,3}, // +Y
        {0,2,3, 0,3,1}, // -Z
        {4,5,7, 4,7,6}  // +Z
    };

    for (int z = 0; z < g->size; z++)
    for (int y = 0; y < g->size; y++)
    for (int x = 0; x < g->size; x++) {
        if (!g->at(x, y, z)) continue;
        Vec3 voxel_color = {1.0f, 1.0f, 1.0f};

        // Build voxel corners
        const Vec3 P[8] = { V(x, y, z), V(x+1, y, z), V(x, y+1, z), V(x+1, y+1, z), V(x, y, z+1), V(x+1, y, z+1), V(x, y+1, z+1), V(x+1, y+1, z+1) };

        // Emit visible faces
        for (int f = 0; f < 6; f++) {
            const int nx = x + offsets[f][0];
            const int ny = y + offsets[f][1];
            const int nz = z + offsets[f][2];
            if (!g->at(nx, ny, nz)) {
                m->transformed_triangles[m->num_triangles++] = { P[faces[f][0]], P[faces[f][1]], P[faces[f][2]], voxel_color };
                m->transformed_triangles[m->num_triangles++] = { P[faces[f][3]], P[faces[f][4]], P[faces[f][5]], voxel_color };
            }
        }
    }

    // Safety check
    assert(m->num_triangles == tri_count);
}

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
    state.voxels.setRandomNoiseSponge();

    memset(&state.voxelModel, 0, sizeof(state.voxelModel));
    buildVoxelModel(&state.voxelModel, &state.voxels);

    renderInit(&state.r, &state.win, &state.cam);

    state.r.light_dir = vec3(0.3f, -1.0f, 0.5f);
    state.running = true;
    state.light_rot = true;

    while (state.running)
    {
        {
            if (pollEvents(&state.win, &state.input)) {
                state.running = false; break;
            }

            if (isKeyDown(&state.input, KEY_LCTRL)) releaseMouse(state.win.window, &state.input);
            else if (!isMouseGrabbed(&state.input)) grabMouse(state.win.window, state.win.width, state.win.height, &state.input);

            state.faster = isKeyDown(&state.input, KEY_LSHIFT);
            const float speed = state.faster ? 4.0f : 2.0f;

            int dx, dy;
            getMouseDelta(&state.input, &dx, &dy);
            cameraRotate(&state.cam, dx * 0.3f, -dy * 0.3f);

            if (isKeyDown(&state.input, KEY_W)) cameraMove(&state.cam, state.cam.front, speed);
            if (isKeyDown(&state.input, KEY_S)) cameraMove(&state.cam, mul(state.cam.front, -1), speed);
            if (isKeyDown(&state.input, KEY_A)) cameraMove(&state.cam, mul(state.cam.right, -1), speed);
            if (isKeyDown(&state.input, KEY_D)) cameraMove(&state.cam, state.cam.right, speed);
        }
        {
            if (state.light_rot)
            {
                static float lightAngle = 0.0f;
                lightAngle += getDelta(&state.win) * 0.2f;
                state.r.light_dir = norm(vec3(-cosf(lightAngle), -0.35f, -sinf(lightAngle)));
            }
            renderClear(&state.r);
            renderModel(&state.r, &state.voxelModel);

            ASSERT(updateFramebuffer(&state.win, state.texture));

            imguiNewFrame();
                ImGui::Begin("voxely");
                ImGui::Text("Pos: %.1f, %.1f, %.1f", state.cam.position.x, state.cam.position.y, state.cam.position.z);
                ImGui::Text("FPS: %.1f (%.2fms)", getFPS(&state.win), getDelta(&state.win) * 1000);
                ImGui::Text("Grid: %dx%dx%d", GRID_SIZE, GRID_SIZE, GRID_SIZE);
                ImGui::Text("Tris: %d", state.voxelModel.num_triangles);
                ImGui::Separator();
                ImGui::Checkbox("Close", &state.running);
                ImGui::Checkbox("Light", &state.r.light);
                ImGui::Checkbox("Light rotate", &state.light_rot);
                ImGui::End();
            imguiEndFrame(&state.win);

            SDL_RenderPresent(state.win.renderer);
            updateFrame(&state.win);
        }
    }

    // Cleanup
    renderFree(&state.r);

    if (state.voxelModel.transformed_triangles) {
        free(state.voxelModel.transformed_triangles);
        state.voxelModel.transformed_triangles = nullptr;
        state.voxelModel.num_triangles = 0;
    }

    SDL_DestroyTexture(state.texture);
    destroyWindow(&state.win);
    return 0;
}
