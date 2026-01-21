#ifndef S64TREE_H
#define S64TREE_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>

typedef struct {
    Vec3 origin;
    Vec3 dir;
    Vec3 inv_dir;
} S64Ray;

typedef struct {
    bool hit;
    float t;
    Vec3 point;
    Vec3 normal;
    uint8_t voxel_id;
} S64Hit;

typedef struct {
    uint32_t is_leaf;
    uint32_t child_ptr;
    uint64_t child_mask;
} S64Node;

typedef struct {
    std::vector<S64Node> node_pool;
    std::vector<uint8_t> leaf_data;
    float bmin[3];
    float bmax[3];
    int grid_size;
} S64Tree;

static inline uint32_t s64_popcnt64(const uint64_t v)
{
    return static_cast<uint32_t>(__builtin_popcountll(v));
}

static inline uint32_t s64_popcnt64_before(const uint64_t mask, const uint32_t idx)
{
    if (idx == 0) return 0;
    return s64_popcnt64(mask & ((1ull << idx) - 1ull));
}

static inline bool s64_aabb_hit(const Vec3 o, const Vec3 invd, const Vec3 bmin, const Vec3 bmax, float* t0, float* t1)
{
    const float tx0 = (bmin.x - o.x) * invd.x;
    const float tx1 = (bmax.x - o.x) * invd.x;
    const float ty0 = (bmin.y - o.y) * invd.y;
    const float ty1 = (bmax.y - o.y) * invd.y;
    const float tz0 = (bmin.z - o.z) * invd.z;
    const float tz1 = (bmax.z - o.z) * invd.z;

    const float tmin = std::max(std::max(std::min(tx0, tx1), std::min(ty0, ty1)), std::min(tz0, tz1));
    const float tmax = std::min(std::min(std::max(tx0, tx1), std::max(ty0, ty1)), std::max(tz0, tz1));

    *t0 = tmin;
    *t1 = tmax;
    return tmax >= tmin;
}

static inline bool s64_block_any(const uint8_t* voxels, const int N, const int x0, const int y0, const int z0, const int size)
{
    for (int z = z0; z < z0 + size; z++)
        for (int y = y0; y < y0 + size; y++)
            for (int x = x0; x < x0 + size; x++)
                if (voxels[(z * N + y) * N + x] != 0) return true;
    return false;
}

static inline uint64_t s64_leaf_mask_and_pack(const uint8_t* voxels, const int N, const int x0, const int y0, const int z0, std::vector<uint8_t>& out)
{
    uint64_t mask = 0;
    for (int y = 0; y < 4; y++)
        for (int z = 0; z < 4; z++)
            for (int x = 0; x < 4; x++) {
                const int idx = x + z * 4 + y * 16;
                if (voxels[((z0 + z) * N + (y0 + y)) * N + (x0 + x)]) mask |= 1ull << idx;
            }

    for (int i = 0; i < 64; i++) {
        if (mask & (1ull << i)) {
            const int x = (i & 3);
            const int z = (i >> 2) & 3;
            const int y = (i >> 4) & 3;
            out.push_back(voxels[((z0 + z) * N + (y0 + y)) * N + (x0 + x)]);
        }
    }
    return mask;
}

static inline uint32_t s64_alloc_node(S64Tree* t) {
    t->node_pool.push_back({0, 0, 0});
    return static_cast<uint32_t>(t->node_pool.size() - 1);
}

static inline bool s64_build_inplace(S64Tree* t, const uint32_t node_idx, const uint8_t* voxels, const int N, const int x0, const int y0, const int z0, const int size)
{
    if (size == 4) {
        const uint64_t ptr = static_cast<uint32_t>(t->leaf_data.size());
        const uint64_t mask = s64_leaf_mask_and_pack(voxels, N, x0, y0, z0, t->leaf_data);
        t->node_pool[node_idx].is_leaf = 1;
        t->node_pool[node_idx].child_ptr = ptr;
        t->node_pool[node_idx].child_mask = mask;
        return mask != 0;
    }

    const int child_size = size / 4;

    uint64_t mask = 0;
    for (int y = 0; y < 4; y++)
        for (int z = 0; z < 4; z++)
            for (int x = 0; x < 4; x++) {
                const int ci = x + z * 4 + y * 16;
                const int cx0 = x0 + x * child_size;
                const int cy0 = y0 + y * child_size;
                const int cz0 = z0 + z * child_size;
                if (s64_block_any(voxels, N, cx0, cy0, cz0, child_size)) mask |= 1ull << ci;
            }

    if (mask == 0) {
        t->node_pool[node_idx] = {0, 0, 0};
        return false;
    }

    const auto first_child = static_cast<uint32_t>(t->node_pool.size());
    const auto child_count = s64_popcnt64(mask);
    for (uint32_t i = 0; i < child_count; i++) s64_alloc_node(t);

    t->node_pool[node_idx].is_leaf = 0;
    t->node_pool[node_idx].child_ptr = first_child;
    t->node_pool[node_idx].child_mask = mask;

    uint32_t slot = 0;
    for (int y = 0; y < 4; y++)
        for (int z = 0; z < 4; z++)
            for (int x = 0; x < 4; x++) {
                if (const int ci = x + z * 4 + y * 16; (mask & (1ull << ci)) == 0) continue;

                const int cx0 = x0 + x * child_size;
                const int cy0 = y0 + y * child_size;
                const int cz0 = z0 + z * child_size;

                s64_build_inplace(t, first_child + slot, voxels, N, cx0, cy0, cz0, child_size);
                slot++;
            }

    return true;
}

static inline S64Tree s64tree_build(const uint8_t* voxels, const int grid_size, const float bounds6[6])
{
    S64Tree t;
    t.node_pool.clear();
    t.leaf_data.clear();
    t.grid_size = grid_size;

    t.bmin[0] = bounds6[0]; t.bmin[1] = bounds6[1]; t.bmin[2] = bounds6[2];
    t.bmax[0] = bounds6[3]; t.bmax[1] = bounds6[4]; t.bmax[2] = bounds6[5];

    t.node_pool.reserve(1024);
    t.leaf_data.reserve(1024);

    s64_alloc_node(&t);

    if (!s64_build_inplace(&t, 0, voxels, grid_size, 0, 0, 0, grid_size)) t.node_pool[0] = {1, 0, 0};

    return t;
}

static inline void s64tree_free(S64Tree* t)
{
    t->node_pool.clear();
    t->leaf_data.clear();
    t->grid_size = 0;
}

static inline bool s64tree_intersect(const S64Tree* t, const S64Ray* ray, S64Hit* out)
{
    out->hit = false;
    out->t = 1e30f;
    out->voxel_id = 0;

    if (t->node_pool.empty()) return false;

    const Vec3 bmin = vec3(t->bmin[0], t->bmin[1], t->bmin[2]);
    const Vec3 bmax = vec3(t->bmax[0], t->bmax[1], t->bmax[2]);

    float t0, t1;
    if (!s64_aabb_hit(ray->origin, ray->inv_dir, bmin, bmax, &t0, &t1)) return false;
    if (t1 < 0.0f) return false;

    float tcur = std::max(t0, 0.0f);
    Vec3 pos = add(ray->origin, mul(ray->dir, tcur));

    for (int iter = 0; iter < 512; iter++)
    {
        uint32_t node_idx = 0;
        S64Node node = t->node_pool[0];

        Vec3 cell_min = bmin;
        float cell_size = (bmax.x - bmin.x);

        bool missing = false;

        while (!node.is_leaf) {
            const float child_size = cell_size * 0.25f;

            int cx = static_cast<int>(std::floor((pos.x - cell_min.x) / child_size));
            int cy = static_cast<int>(std::floor((pos.y - cell_min.y) / child_size));
            int cz = static_cast<int>(std::floor((pos.z - cell_min.z) / child_size));

            cx = std::clamp(cx, 0, 3);
            cy = std::clamp(cy, 0, 3);
            cz = std::clamp(cz, 0, 3);

            const auto child_idx = static_cast<uint32_t>(cx + cz * 4 + cy * 16);

            if ((node.child_mask & (1ull << child_idx)) == 0)
            {
                cell_min = add(cell_min, vec3(cx * child_size, cy * child_size, cz * child_size));
                cell_size = child_size;
                missing = true;
                break;
            }

            const uint32_t slot = s64_popcnt64_before(node.child_mask, child_idx);
            node_idx = node.child_ptr + slot;
            if (node_idx >= t->node_pool.size()) return false;
            node = t->node_pool[node_idx];

            cell_min = add(cell_min, vec3(cx * child_size, cy * child_size, cz * child_size));
            cell_size = child_size;
        }

        if (node.is_leaf && !missing) {
            const float voxel_size = cell_size * 0.25f;

            int vx = static_cast<int>(std::floor((pos.x - cell_min.x) / voxel_size));
            int vy = static_cast<int>(std::floor((pos.y - cell_min.y) / voxel_size));
            int vz = static_cast<int>(std::floor((pos.z - cell_min.z) / voxel_size));

            vx = std::clamp(vx, 0, 3);
            vy = std::clamp(vy, 0, 3);
            vz = std::clamp(vz, 0, 3);

            if (const auto vidx = static_cast<uint32_t>(vx + vz * 4 + vy * 16); node.child_mask & (1ull << vidx))
            {
                const uint32_t slot = s64_popcnt64_before(node.child_mask, vidx);
                const uint32_t lp = node.child_ptr + slot;
                if (lp >= t->leaf_data.size()) return false;

                out->hit = true;
                out->t = tcur;
                out->point = pos;
                out->normal = vec3(0, 1, 0);
                out->voxel_id = t->leaf_data[lp];
                return true;
            }
        }

        auto [x, y, z] = cell_min;
        auto [x1, y1, z1] = add(cell_min, vec3(cell_size, cell_size, cell_size));

        float tx = (ray->dir.x > 0.0f) ? (x1 - pos.x) / ray->dir.x : (ray->dir.x < 0.0f ? (x - pos.x) / ray->dir.x : 1e30f);
        float ty = (ray->dir.y > 0.0f) ? (y1 - pos.y) / ray->dir.y : (ray->dir.y < 0.0f ? (y - pos.y) / ray->dir.y : 1e30f);
        float tz = (ray->dir.z > 0.0f) ? (z1 - pos.z) / ray->dir.z : (ray->dir.z < 0.0f ? (z - pos.z) / ray->dir.z : 1e30f);

        float dt = std::min(tx, std::min(ty, tz));
        if (!(dt > 0.0f)) dt = 1e-4f;

        tcur += dt + 1e-4f;
        if (tcur > t1) break;

        pos = add(ray->origin, mul(ray->dir, tcur));
    }

    return false;
}

#endif
