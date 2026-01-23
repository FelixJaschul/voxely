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

#define GRID_SIZE 40
#define WIDTH 1250
#define HEIGHT 850

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

    void sierpinskiRec(const int x, const int y, const int z, const int size)
    {
        if (size <= 0) return;
        if (size == 1) {
            data[z][y][x] = 1;
            return;
        }

        for (int dz = 0; dz < 3; dz++)
        for (int dy = 0; dy < 3; dy++)
        for (int dx = 0; dx < 3; dx++) {
            if (dx == 1 && dy == 1 && dz == 1) continue;
            sierpinskiRec(
                x + dx * size / 3,
                y + dy * size / 3,
                z + dz * size / 3,
                size / 3
            );
        }
    }

    void setSierpinski()
    {
        memset(data, 0, sizeof(data));
        sierpinskiRec(0, 0, 0, size);
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
    auto V = [&](const float x, const float y, const float z) {
        return vec3(x - g->size * 0.5f, y - g->size * 0.5f, z - g->size * 0.5f);
    };

    auto VEC = [&](float h, const float s, const float v) {
        h = fmodf(h, 360.0f);
        if (h < 0) h += 360.0f;
        const float c = v * s;
        const float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
        const float m = v - c;
        float r, g, b;
        if (h < 60)       { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else              { r = c; g = 0; b = x; }
        return vec3(r + m, g + m, b + m);
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

    // Allocate exactly what we need
    m->transformed_triangles = static_cast<Triangle*>(malloc(sizeof(Triangle) * tri_count));
    m->num_triangles = 0;

    // Precompute normalization factor
    const float invSize = (g->size > 1) ? 1.0f / (g->size - 1) : 1.0f;

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

        // Compute diagonal rainbow color
        const float t = (x * invSize - z * invSize + 1.0f) * 0.5f;
        const Vec3 voxel_color = VEC(fmaxf(0.0f, fminf(1.0f, t)) * 360.0f, 1.0f, 1.0f);

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
    state.voxels.setSierpinski();

    memset(&state.voxelModel, 0, sizeof(state.voxelModel));
    buildVoxelModel(&state.voxelModel, &state.voxels);

    renderInit(&state.r, &state.win, &state.cam);

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
            static float lightAngle = 0.0f;
            lightAngle += getDelta(&state.win) * 0.2f;
            state.r.light_dir = norm(vec3(-cosf(lightAngle), -0.35f, -sinf(lightAngle)));
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
            ImGui::Checkbox("Light", &state.r.light);
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
