// ============================================================================
// Bounding Volume Hierarchy for fast ray-triangle intersection
// ============================================================================
#ifndef BVH_H
#define BVH_H

#include <cfloat>
#include <stdlib.h>
#include <math.h>

#ifndef EPSILON
#define EPSILON 0.0001f
#endif

#define STACK_SIZE 1028
#define LEAF_SIZE 4

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { Vec3 min, max; } AABB;

typedef struct BVHNode {
    AABB bounds;
    struct BVHNode *left, *right;
    Triangle *tris;
    Material *mats;
    int count;
} BVHNode;

struct BvhRay {
    Vec3 origin, direction, inv_direction;
};

struct HitRecord {
    bool hit;
    float t;
    Vec3 point, normal;
    Material mat;
};

static inline void bvh_build(BVHNode **root, const Model *models, int num);
void bvh_free(BVHNode *n);
bool bvh_intersect(BVHNode *root, const BvhRay &ray, HitRecord *rec);

#ifdef __cplusplus
}
#endif

#ifdef BVH_IMPLEMENTATION

typedef struct {
    Triangle tri;
    Material mat;
    Vec3 center;
} Item;

static inline AABB box_tri(const Triangle &t)
{
    AABB b;
    b.min.x = fminf(fminf(t.v0.x, t.v1.x), t.v2.x);
    b.min.y = fminf(fminf(t.v0.y, t.v1.y), t.v2.y);
    b.min.z = fminf(fminf(t.v0.z, t.v1.z), t.v2.z);
    b.max.x = fmaxf(fmaxf(t.v0.x, t.v1.x), t.v2.x);
    b.max.y = fmaxf(fmaxf(t.v0.y, t.v1.y), t.v2.y);
    b.max.z = fmaxf(fmaxf(t.v0.z, t.v1.z), t.v2.z);
    return b;
}

static inline AABB box_merge(const AABB &a, const AABB &b)
{
    AABB r;
    r.min.x = fminf(a.min.x, b.min.x); r.min.y = fminf(a.min.y, b.min.y); r.min.z = fminf(a.min.z, b.min.z);
    r.max.x = fmaxf(a.max.x, b.max.x); r.max.y = fmaxf(a.max.y, b.max.y); r.max.z = fmaxf(a.max.z, b.max.z);
    return r;
}

static inline float box_surface_area(const AABB &box)
{
    const float dx = box.max.x - box.min.x;
    const float dy = box.max.y - box.min.y;
    const float dz = box.max.z - box.min.z;
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

static inline bool box_hit(const AABB &box, const BvhRay &r, float tmin, float tmax)
{
    const float t0x = (box.min.x - r.origin.x) * r.inv_direction.x;
    const float t1x = (box.max.x - r.origin.x) * r.inv_direction.x;
    const float t0y = (box.min.y - r.origin.y) * r.inv_direction.y;
    const float t1y = (box.max.y - r.origin.y) * r.inv_direction.y;
    const float t0z = (box.min.z - r.origin.z) * r.inv_direction.z;
    const float t1z = (box.max.z - r.origin.z) * r.inv_direction.z;

    tmin = fmaxf(tmin, fmaxf(fminf(t0x, t1x), fmaxf(fminf(t0y, t1y), fminf(t0z, t1z))));
    tmax = fminf(tmax, fminf(fmaxf(t0x, t1x), fminf(fmaxf(t0y, t1y), fmaxf(t0z, t1z))));

    return tmax >= tmin;
}

static int cmp_x(const void *a, const void *b) {
    return (((Item*)a)->center.x > ((Item*)b)->center.x) - (((Item*)a)->center.x < ((Item*)b)->center.x);
}
static int cmp_y(const void *a, const void *b) {
    return (((Item*)a)->center.y > ((Item*)b)->center.y) - (((Item*)a)->center.y < ((Item*)b)->center.y);
}
static int cmp_z(const void *a, const void *b) {
    return (((Item*)a)->center.z > ((Item*)b)->center.z) - (((Item*)a)->center.z < ((Item*)b)->center.z);
}

static BVHNode* build(Item *items, const int n)
{
    const auto node = static_cast<BVHNode *>(malloc(sizeof(BVHNode)));
    node->bounds = box_tri(items[0].tri);
    for (int i = 1; i < n; i++)
        node->bounds = box_merge(node->bounds, box_tri(items[i].tri));

    if (n <= LEAF_SIZE) {
        node->count = n;
        node->tris = static_cast<Triangle *>(malloc(n * sizeof(Triangle)));
        node->mats = static_cast<Material *>(malloc(n * sizeof(Material)));
        for (int i = 0; i < n; i++) {
            node->tris[i] = items[i].tri;
            node->mats[i] = items[i].mat;
        }
        node->left = node->right = NULL;
        return node;
    }

    const Vec3 extent = sub(node->bounds.max, node->bounds.min);
    int axis = (extent.y > extent.x) ? 1 : 0;
    if (extent.z <= ((float *) &extent)[axis]) axis = 2;

    qsort(items, n, sizeof(Item), axis == 0 ? cmp_x : axis == 1 ? cmp_y : cmp_z);

    float best_cost = FLT_MAX;
    int best_split = n / 2;
    const int num_buckets = (n < 16) ? n : 16;

    for (int i = 1; i < num_buckets && i < n; i++) {
        const int split = (n * i) / num_buckets;
        if (split <= 0 || split >= n) continue;

        AABB left_box = box_tri(items[0].tri);
        for (int j = 1; j < split; j++) left_box = box_merge(left_box, box_tri(items[j].tri));

        AABB right_box = box_tri(items[split].tri);
        for (int j = split + 1; j < n; j++) right_box = box_merge(right_box, box_tri(items[j].tri));

        if (const float cost = box_surface_area(left_box) * split + box_surface_area(right_box) * (n - split); cost < best_cost) {
            best_cost = cost;
            best_split = split;
        }
    }

    node->count = 0;
    node->tris = nullptr;
    node->mats = nullptr;
    node->left = build(items, best_split);
    node->right = build(items + best_split, n - best_split);
    return node;
}

static inline void bvh_build(BVHNode **root, const Model *models, const int num)
{
    int total = 0;
    for (int i = 0; i < num; i++) total += models[i].num_triangles;

    const auto items = static_cast<Item *>(malloc(total * sizeof(Item)));
    int idx = 0;
    for (int i = 0; i < num; i++) {
        const Model *m = &models[i];
        for (int j = 0; j < m->num_triangles; j++) {
            const Triangle t = m->transformed_triangles[j];
            items[idx].tri = t;
            items[idx].mat = m->mat;
            items[idx].center = vec3((t.v0.x + t.v1.x + t.v2.x) / 3.0f,
                                     (t.v0.y + t.v1.y + t.v2.y) / 3.0f,
                                     (t.v0.z + t.v1.z + t.v2.z) / 3.0f);
            idx++;
        }
    }

    *root = build(items, total);
    free(items);
}

inline void bvh_free(BVHNode *n) {
    if (!n) return;
    if (n->count) {
        free(n->tris);
        free(n->mats);
    } else {
        bvh_free(n->left);
        bvh_free(n->right);
    }
    free(n);
}

static inline bool intersect_triangle(const BvhRay &ray, const Triangle &tri, const Material &mat, HitRecord *rec)
{
    const Vec3 edge1 = sub(tri.v1, tri.v0);
    const Vec3 edge2 = sub(tri.v2, tri.v0);
    const Vec3 h = cross(ray.direction, edge2);
    const float a = dot(edge1, h);

    if (fabsf(a) < 1e-6f) return false;

    const float f = 1.0f / a;
    const Vec3 s = sub(ray.origin, tri.v0);
    const float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    const Vec3 q = cross(s, edge1);

    if (const float v = f * dot(ray.direction, q); v < 0.0f || u + v > 1.0f) return false;

    const float t = f * dot(edge2, q);

    if (t < EPSILON || t >= rec->t) return false;

    rec->hit = true;
    rec->t = t;
    rec->point = add(ray.origin, mul(ray.direction, t));
    rec->normal = norm(cross(edge1, edge2));
    rec->mat = mat;

    return true;
}

inline bool bvh_intersect(BVHNode *root, const BvhRay &ray, HitRecord *rec)
{
    if (!root) return false;
    BVHNode *stack[STACK_SIZE];
    int sp = 0;
    stack[sp++] = root;
    bool hit = false;

    while (sp > 0) {
        const BVHNode *n = stack[--sp];
        if (!box_hit(n->bounds, ray, 0.001f, rec->t)) continue;

        if (n->count) {
            for (int i = 0; i < n->count; i++)
                if (intersect_triangle(ray, n->tris[i], n->mats[i], rec))
                    hit = true;
        } else {
            if (sp + 2 < STACK_SIZE) {
                stack[sp++] = n->left;
                stack[sp++] = n->right;
            }
        }
    }
    return hit;
}

#endif // BVH_IMPLEMENTATION
#endif // BVH_H