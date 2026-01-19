// ============================================================================
// core.h - Core window management (X11 or SDL3 backend)
// ============================================================================
#ifndef WRAPPER_CORE_H
#define WRAPPER_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SDL_IMPLEMENTATION
    #include <SDL3/SDL.h>
#else
    #include <X11/Xlib.h>
    #include <X11/Xutil.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Window {
#ifdef SDL_IMPLEMENTATION
    SDL_Window   *window;
    SDL_Renderer *renderer;
#else
    Window   window;
    Display *display;
    XImage  *image;
    int      screen;
    GC       gc;
#endif

    int width;
    int height;
    int x, y;
    const char *title;
    uint32_t *buffer;    // Direct pixel buffer for software rendering
    double fps;          // Target frames per second
    double deltat;       // Delta time of last frame in seconds
    struct timespec lastt;
} Window_t;

// Initialize window structure with default values
/*  -> Example:
 *  Window_t win;
 *  windowInit(&win);
 *  win.title = "My Window";
 *  win.width = 1280;
 *  win.height = 720;
 */
void windowInit(Window_t *w);

// Create and display window, returns false on failure
/*  -> Example:
 *  if (!createWindow(&win)) return 1;
 */
bool createWindow(Window_t *w);

// Cleanup and destroy window resources
/*  -> Example:
 *  destroyWindow(&win);
 */
void destroyWindow(Window_t *w);

// Update frame timing (sleeps to maintain target FPS)
/*  -> Example:
 *  while(1) {
 *      ...
 *      updateFrame(&win);
 *  }
 */
void updateFrame(Window_t *w);

// Push pixel buffer to screen
/*  -> Example:
 *  // After drawing to win.buffer
 *  updateFramebuffer(&win);
 */
void updateFramebuffer(const Window_t *w);

// Get current FPS based on actual frame time
/*  -> Example:
 *  printf("FPS: %.2f\n", getFPS(&win));
 */
double getFPS(const Window_t *w);

// Draw single pixel at (x,y) with bounds checking
/*  -> Example:
 *  drawPixel(&win, 100, 100, 0xFF0000); // Draw red pixel
 */
void drawPixel(const Window_t *w, int x, int y, uint32_t color);

#ifdef __cplusplus
}
#endif

// ============================================================================
// IMPLEMENTATION
// ============================================================================

#ifdef CORE_IMPLEMENTATION

inline void windowInit(Window_t *w)
{
#ifdef SDL_IMPLEMENTATION
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    }
    w->window   = NULL;
    w->renderer = NULL;
#else
    w->display = XOpenDisplay(NULL);
    w->screen  = 0;
    w->window  = 0;
    w->gc      = 0;
    w->image   = NULL;
#endif

    w->width  = 800;
    w->height = 600;
    w->x      = 100;
    w->y      = 100;
    w->title  = "DEMO WINDOW";
    w->buffer = NULL;
    w->fps    = 60.0;
    w->deltat = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &w->lastt);
}

inline bool createWindow(Window_t *w)
{
#ifdef SDL_IMPLEMENTATION
    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
            return false;
        }
    }

    w->window = SDL_CreateWindow(
        w->title,
        w->width,
        w->height,
        SDL_WINDOW_RESIZABLE
    );
    if (!w->window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }

    w->renderer = SDL_CreateRenderer(w->window, NULL, SDL_RENDERER_ACCELERATED);
    if (!w->renderer) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(w->window);
        w->window = NULL;
        return false;
    }

    w->buffer = (uint32_t*)malloc(w->width * w->height * sizeof(uint32_t));
    assert(w->buffer && "Failed to allocate framebuffer");

    clock_gettime(CLOCK_MONOTONIC, &w->lastt);
    return true;

#else
    if (!w->display) {
        fprintf(stderr, "Failed to open X11 display\n");
        return false;
    }

    w->window = XCreateSimpleWindow(
        w->display, RootWindow(w->display, w->screen),
        w->x, w->y, w->width, w->height, 0,
        BlackPixel(w->display, w->screen),
        WhitePixel(w->display, w->screen)
    );

    if (!w->window) {
        fprintf(stderr, "Failed to create X11 window\n");
        return false;
    }

    XStoreName(w->display, w->window, w->title);
    XSelectInput(w->display, w->window,
        ExposureMask | KeyPressMask | KeyReleaseMask |
        StructureNotifyMask | PointerMotionMask |
        ButtonPressMask | ButtonReleaseMask);

    Atom wmDeleteMessage = XInternAtom(w->display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(w->display, w->window, &wmDeleteMessage, 1);

    XMapWindow(w->display, w->window);
    XFlush(w->display);

    XEvent ev;
    XWindowEvent(w->display, w->window, ExposureMask, &ev);

    w->gc = DefaultGC(w->display, w->screen);
    w->buffer = (uint32_t *) malloc(w->width * w->height * sizeof(uint32_t));
    assert(w->buffer && "Failed to allocate framebuffer");

    w->image = XCreateImage(
        w->display,
        DefaultVisual(w->display, w->screen),
        DefaultDepth(w->display, w->screen),
        ZPixmap, 0, (char*) w->buffer,
        w->width, w->height, 32, 0
    );
    assert(w->image && "Failed to create XImage");

    clock_gettime(CLOCK_MONOTONIC, &w->lastt);
    return true;
#endif
}

inline void destroyWindow(Window_t *w)
{
#ifdef SDL_IMPLEMENTATION
    if (w->renderer) {
        SDL_DestroyRenderer(w->renderer);
        w->renderer = NULL;
    }
    if (w->window) {
        SDL_DestroyWindow(w->window);
        w->window = NULL;
    }
    if (w->buffer) {
        free(w->buffer);
        w->buffer = NULL;
    }
    SDL_Quit();
#else
    if (!w->display) return;

    if (w->image) {
        XDestroyImage(w->image);
        w->image  = NULL;
        w->buffer = NULL;
    } else if (w->buffer) {
        free(w->buffer);
        w->buffer = NULL;
    }

    if (w->window) {
        XDestroyWindow(w->display, w->window);
        w->window = 0;
    }

    XSync(w->display, False);
    XCloseDisplay(w->display);
    w->display = NULL;
#endif
}

inline void updateFrame(Window_t *w)
{
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    double elapsed = (current_time.tv_sec - w->lastt.tv_sec) +
                     (current_time.tv_nsec - w->lastt.tv_nsec) / 1e9;

    const double target_frame_time = 1.0 / w->fps;
    if (elapsed < target_frame_time) {
        const double sleep_time = target_frame_time - elapsed;
        struct timespec sleep_spec;
        sleep_spec.tv_sec = (time_t)sleep_time;
        sleep_spec.tv_nsec = (long)((sleep_time - sleep_spec.tv_sec) * 1e9);
        nanosleep(&sleep_spec, NULL);

        clock_gettime(CLOCK_MONOTONIC, &current_time);
        elapsed = (current_time.tv_sec - w->lastt.tv_sec) +
                  (current_time.tv_nsec - w->lastt.tv_nsec) / 1e9;
    }

    w->deltat = elapsed;
    w->lastt  = current_time;
}

inline void updateFramebuffer(const Window_t *w)
{
#ifdef SDL_IMPLEMENTATION
    if (!w->renderer || !w->buffer) return;

    SDL_Texture *tex = SDL_CreateTexture(
        w->renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        w->width,
        w->height
    );
    if (!tex) return;

    void *pixels = NULL;
    int pitch   = 0;
    if (SDL_LockTexture(tex, NULL, &pixels, &pitch) == 0) {
        uint8_t *dst = (uint8_t*)pixels;
        const uint8_t *src = (const uint8_t*)w->buffer;
        const int row_bytes = w->width * 4;
        for (int y = 0; y < w->height; ++y) {
            memcpy(dst + y * pitch, src + y * row_bytes, row_bytes);
        }
        SDL_UnlockTexture(tex);
    }

    SDL_RenderClear(w->renderer);
    SDL_RenderTexture(w->renderer, tex, NULL, NULL);
    SDL_RenderPresent(w->renderer);

    SDL_DestroyTexture(tex);
#else
    XPutImage(w->display, w->window, w->gc, w->image,
              0, 0, 0, 0, w->width, w->height);
    XFlush(w->display);
#endif
}

inline double getFPS(const Window_t *w)
{
    return (w->deltat > 0.0) ? (1.0 / w->deltat) : 0.0;
}

inline void drawPixel(const Window_t *w, int x, int y, uint32_t color)
{
    if (x >= 0 && x < w->width && y >= 0 && y < w->height)
        w->buffer[y * w->width + x] = color;
}

#endif // CORE_IMPLEMENTATION
#endif // WRAPPER_CORE_H

// ============================================================================
// keys.h - Keyboard and Mouse Input (X11 or SDL3 backend)
// ============================================================================
#ifndef WRAPPER_KEYS_H
#define WRAPPER_KEYS_H

#include <stdbool.h>

#ifdef SDL_IMPLEMENTATION
    #include <SDL3/SDL.h>
#else
    #include <X11/Xlib.h>
    #include <X11/keysym.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KEY_UNKNOWN = 0,

    KEY_ESCAPE, KEY_SPACE,
    KEY_ENTER,  KEY_TAB, KEY_BACKSPACE,

    KEY_LEFT, KEY_RIGHT,
    KEY_UP,   KEY_DOWN,

    KEY_A, KEY_B, KEY_C,
    KEY_D, KEY_E, KEY_F,
    KEY_G, KEY_H, KEY_I,
    KEY_J, KEY_K, KEY_L,
    KEY_M, KEY_N, KEY_O,
    KEY_P, KEY_Q, KEY_R,
    KEY_S, KEY_T, KEY_U,
    KEY_V, KEY_W, KEY_X,
    KEY_Y, KEY_Z,

    KEY_NUM0, KEY_NUM1, KEY_NUM2,
    KEY_NUM3, KEY_NUM4, KEY_NUM5,
    KEY_NUM6, KEY_NUM7, KEY_NUM8,
    KEY_NUM9,

    KEY_LSHIFT, KEY_RSHIFT,
    KEY_LCTRL,  KEY_RCTRL,
    KEY_LALT,   KEY_RALT,

    KEY_F1, KEY_F2,
    KEY_F3, KEY_F4,

    KEY_COUNT
} Key;

typedef enum {
    MOUSE_LEFT = 0,
    MOUSE_MIDDLE,
    MOUSE_RIGHT,
    MOUSE_COUNT
} MouseButton;

// Input state structure - must be initialized and passed to all functions
typedef struct InputState {
    bool key_curr[KEY_COUNT];
    bool key_prev[KEY_COUNT];
    bool mouse_curr[MOUSE_COUNT];
    bool mouse_prev[MOUSE_COUNT];
    int mouse_x, mouse_y;
    int mouse_dx, mouse_dy;
    int last_x, last_y;
    bool mouse_grabbed;
#ifdef SDL_IMPLEMENTATION
    void *grab_window;
#else
    Window grab_window;
#endif
    int center_x, center_y;
} Input;

// Initialize input state to zero
/*  -> Example:
 *  Input input;
 *  inputInit(&input);
 */
void inputInit(Input *input);

// Poll events and update input state, returns true if window should close
/*  -> Example:
 *  #ifdef SDL_IMPLEMENTATION
 *  if (pollEvents(NULL, &input)) break;
 *  #else
 *  if (pollEvents(win.display, &input)) break;
 *  #endif
 */
#ifdef SDL_IMPLEMENTATION
bool pollEvents(void *display, Input *input);
#else
bool pollEvents(Display *display, Input *input);
#endif

// Copy current frame state to previous (call once per frame after processing)
/*  -> Example:
 *  while(1) {
 *      ...
 *      updateInput(&input);
 *  }
 */
void updateInput(Input *input);

// Check if key is currently held down
/*  -> Example:
 *  if (isKeyDown(&input, KEY_W)) // Move forward
 */
bool isKeyDown(const Input *input, Key key);

// Check if key was just pressed this frame
/*  -> Example:
 *  if (isKeyPressed(&input, KEY_SPACE)) // Jump
 */
bool isKeyPressed(const Input *input, Key key);

// Check if key was just released this frame
/*  -> Example:
 *  if (isKeyReleased(&input, KEY_E)) // Use item
 */
bool isKeyReleased(const Input *input, Key key);

// Check if mouse button is currently held down
/*  -> Example:
 *  if (isMouseDown(&input, MOUSE_LEFT)) // Shoot
 */
bool isMouseDown(const Input *input, MouseButton btn);

// Check if mouse button was just pressed this frame
/*  -> Example:
 *  if (isMousePressed(&input, MOUSE_LEFT)) // Fire once
 */
bool isMousePressed(const Input *input, MouseButton btn);

// Check if mouse button was just released this frame
/*  -> Example:
 *  if (isMouseReleased(&input, MOUSE_LEFT)) // Release
 */
bool isMouseReleased(const Input *input, MouseButton btn);

// Get current mouse position in window coordinates
/*  -> Example:
 *  int mx, my;
 *  getMousePosition(&input, &mx, &my);
 */
void getMousePosition(const Input *input, int *x, int *y);

// Get mouse delta since last frame (for FPS camera controls)
/*  -> Example:
 *  int dx, dy;
 *  getMouseDelta(&input, &dx, &dy);
 */
void getMouseDelta(const Input *input, int *dx, int *dy);

// Grab mouse cursor (hides and locks to window center)
/*  -> Example:
 *  #ifdef SDL_IMPLEMENTATION
 *  grabMouse(win.window, win.width, win.height, &input);
 *  #else
 *  grabMouse(win.display, win.window, win.width, win.height, &input);
 *  #endif
 */
#ifdef SDL_IMPLEMENTATION
void grabMouse(SDL_Window *window, int width, int height, Input *input);
#else
void grabMouse(Display *display, Window window, int width, int height, Input *input);
#endif

// Release grabbed mouse cursor
/*  -> Example:
 *  #ifdef SDL_IMPLEMENTATION
 *  releaseMouse(win.window, &input);
 *  #else
 *  releaseMouse(win.display, win.window, &input);
 *  #endif
 */
#ifdef SDL_IMPLEMENTATION
void releaseMouse(SDL_Window *window, Input *input);
#else
void releaseMouse(Display *display, Window window, Input *input);
#endif

// Check if mouse is currently grabbed
/*  -> Example:
 *  if (isMouseGrabbed(&input)) { ... }
 */
bool isMouseGrabbed(const Input *input);

#ifdef __cplusplus
}
#endif

#ifdef KEYS_IMPLEMENTATION

static void setKey(Input *input, const Key key, const bool down)
{
    if (key < KEY_COUNT) input->key_curr[key] = down;
}

static void setMouse(Input *input, const MouseButton btn, const bool down)
{
    if (btn < MOUSE_COUNT) input->mouse_curr[btn] = down;
}

inline void inputInit(Input *input)
{
    for (int i = 0; i < KEY_COUNT; i++) {
        input->key_curr[i] = false;
        input->key_prev[i] = false;
    }
    for (int i = 0; i < MOUSE_COUNT; i++) {
        input->mouse_curr[i] = false;
        input->mouse_prev[i] = false;
    }
    input->mouse_x = 0;
    input->mouse_y = 0;
    input->mouse_dx = 0;
    input->mouse_dy = 0;
    input->last_x = 0;
    input->last_y = 0;
    input->mouse_grabbed = false;
    input->grab_window = 0;
    input->center_x = 400;
    input->center_y = 300;
}

#ifdef SDL_IMPLEMENTATION

inline bool pollEvents(void *display, Input *input)
{
    (void)display;
    bool shouldClose = false;

    input->mouse_dx = 0;
    input->mouse_dy = 0;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                shouldClose = true;
                break;

            case SDL_EVENT_MOUSE_MOTION:
                if (input->mouse_grabbed) {
                    input->mouse_dx += (int)event.motion.xrel;
                    input->mouse_dy += (int)event.motion.yrel;
                }
                input->mouse_x = (int)event.motion.x;
                input->mouse_y = (int)event.motion.y;
                input->last_x  = input->mouse_x;
                input->last_y  = input->mouse_y;
                break;

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                const bool down = (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
                switch (event.button.button) {
                    case SDL_BUTTON_LEFT:   setMouse(input, MOUSE_LEFT,   down); break;
                    case SDL_BUTTON_MIDDLE: setMouse(input, MOUSE_MIDDLE, down); break;
                    case SDL_BUTTON_RIGHT:  setMouse(input, MOUSE_RIGHT,  down); break;
                    default: break;
                }
                break;
            }

            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP: {
                const bool down = (event.type == SDL_EVENT_KEY_DOWN);
                const SDL_Keycode sym = event.key.keysym.sym;
                Key k = KEY_UNKNOWN;

                switch (sym) {
                    case SDLK_ESCAPE:  k = KEY_ESCAPE; break;
                    case SDLK_SPACE:   k = KEY_SPACE; break;
                    case SDLK_RETURN:  k = KEY_ENTER; break;
                    case SDLK_TAB:     k = KEY_TAB; break;
                    case SDLK_BACKSPACE: k = KEY_BACKSPACE; break;

                    case SDLK_LEFT:  k = KEY_LEFT; break;
                    case SDLK_RIGHT: k = KEY_RIGHT; break;
                    case SDLK_UP:    k = KEY_UP; break;
                    case SDLK_DOWN:  k = KEY_DOWN; break;

                    case SDLK_a: k = KEY_A; break;
                    case SDLK_b: k = KEY_B; break;
                    case SDLK_c: k = KEY_C; break;
                    case SDLK_d: k = KEY_D; break;
                    case SDLK_e: k = KEY_E; break;
                    case SDLK_f: k = KEY_F; break;
                    case SDLK_g: k = KEY_G; break;
                    case SDLK_h: k = KEY_H; break;
                    case SDLK_i: k = KEY_I; break;
                    case SDLK_j: k = KEY_J; break;
                    case SDLK_k: k = KEY_K; break;
                    case SDLK_l: k = KEY_L; break;
                    case SDLK_m: k = KEY_M; break;
                    case SDLK_n: k = KEY_N; break;
                    case SDLK_o: k = KEY_O; break;
                    case SDLK_p: k = KEY_P; break;
                    case SDLK_q: k = KEY_Q; break;
                    case SDLK_r: k = KEY_R; break;
                    case SDLK_s: k = KEY_S; break;
                    case SDLK_t: k = KEY_T; break;
                    case SDLK_u: k = KEY_U; break;
                    case SDLK_v: k = KEY_V; break;
                    case SDLK_w: k = KEY_W; break;
                    case SDLK_x: k = KEY_X; break;
                    case SDLK_y: k = KEY_Y; break;
                    case SDLK_z: k = KEY_Z; break;

                    case SDLK_0: k = KEY_NUM0; break;
                    case SDLK_1: k = KEY_NUM1; break;
                    case SDLK_2: k = KEY_NUM2; break;
                    case SDLK_3: k = KEY_NUM3; break;
                    case SDLK_4: k = KEY_NUM4; break;
                    case SDLK_5: k = KEY_NUM5; break;
                    case SDLK_6: k = KEY_NUM6; break;
                    case SDLK_7: k = KEY_NUM7; break;
                    case SDLK_8: k = KEY_NUM8; break;
                    case SDLK_9: k = KEY_NUM9; break;

                    case SDLK_LSHIFT: k = KEY_LSHIFT; break;
                    case SDLK_RSHIFT: k = KEY_RSHIFT; break;
                    case SDLK_LCTRL:  k = KEY_LCTRL; break;
                    case SDLK_RCTRL:  k = KEY_RCTRL; break;
                    case SDLK_LALT:   k = KEY_LALT; break;
                    case SDLK_RALT:   k = KEY_RALT; break;

                    case SDLK_F1: k = KEY_F1; break;
                    case SDLK_F2: k = KEY_F2; break;
                    case SDLK_F3: k = KEY_F3; break;
                    case SDLK_F4: k = KEY_F4; break;

                    default: break;
                }

                if (k != KEY_UNKNOWN) setKey(input, k, down);
                break;
            }

            default:
                break;
        }
    }

    return shouldClose;
}

#else // X11 backend

inline bool pollEvents(Display *display, Input *input)
{
    const Atom wmDeleteMessage = XInternAtom(display, "WM_DELETE_WINDOW", False);
    bool shouldClose = false;

    input->mouse_dx = 0;
    input->mouse_dy = 0;

    while (XPending(display) > 0)
    {
        XEvent event;
        XNextEvent(display, &event);

        switch (event.type)
        {
            case ClientMessage:
                if ((Atom)event.xclient.data.l[0] == wmDeleteMessage)
                    shouldClose = true;
                break;

            case MotionNotify:
                if (input->mouse_grabbed) {
                    input->mouse_dx += event.xmotion.x - input->last_x;
                    input->mouse_dy += event.xmotion.y - input->last_y;
                }
                input->mouse_x = event.xmotion.x;
                input->mouse_y = event.xmotion.y;
                input->last_x = input->mouse_x;
                input->last_y = input->mouse_y;
                break;

            case ButtonPress:
            case ButtonRelease:
            {
                const bool down = (event.type == ButtonPress);
                switch (event.xbutton.button)
                {
                    case Button1: setMouse(input, MOUSE_LEFT, down); break;
                    case Button2: setMouse(input, MOUSE_MIDDLE, down); break;
                    case Button3: setMouse(input, MOUSE_RIGHT, down); break;
                    default: break;
                }
                break;
            }

            case KeyPress:
            case KeyRelease:
            {
                const bool down = (event.type == KeyPress);
                const KeySym sym = XLookupKeysym(&event.xkey, 0);
                Key k = KEY_UNKNOWN;

                switch (sym)
                {
                    case XK_Escape: k = KEY_ESCAPE; break;
                    case XK_space: k = KEY_SPACE; break;
                    case XK_Return: k = KEY_ENTER; break;
                    case XK_Tab: k = KEY_TAB; break;
                    case XK_BackSpace: k = KEY_BACKSPACE; break;

                    case XK_Left: k = KEY_LEFT; break;
                    case XK_Right: k = KEY_RIGHT; break;
                    case XK_Up: k = KEY_UP; break;
                    case XK_Down: k = KEY_DOWN; break;

                    case XK_a: case XK_A: k = KEY_A; break;
                    case XK_b: case XK_B: k = KEY_B; break;
                    case XK_c: case XK_C: k = KEY_C; break;
                    case XK_d: case XK_D: k = KEY_D; break;
                    case XK_e: case XK_E: k = KEY_E; break;
                    case XK_f: case XK_F: k = KEY_F; break;
                    case XK_g: case XK_G: k = KEY_G; break;
                    case XK_h: case XK_H: k = KEY_H; break;
                    case XK_i: case XK_I: k = KEY_I; break;
                    case XK_j: case XK_J: k = KEY_J; break;
                    case XK_k: case XK_K: k = KEY_K; break;
                    case XK_l: case XK_L: k = KEY_L; break;
                    case XK_m: case XK_M: k = KEY_M; break;
                    case XK_n: case XK_N: k = KEY_N; break;
                    case XK_o: case XK_O: k = KEY_O; break;
                    case XK_p: case XK_P: k = KEY_P; break;
                    case XK_q: case XK_Q: k = KEY_Q; break;
                    case XK_r: case XK_R: k = KEY_R; break;
                    case XK_s: case XK_S: k = KEY_S; break;
                    case XK_t: case XK_T: k = KEY_T; break;
                    case XK_u: case XK_U: k = KEY_U; break;
                    case XK_v: case XK_V: k = KEY_V; break;
                    case XK_w: case XK_W: k = KEY_W; break;
                    case XK_x: case XK_X: k = KEY_X; break;
                    case XK_y: case XK_Y: k = KEY_Y; break;
                    case XK_z: case XK_Z: k = KEY_Z; break;

                    case XK_0: k = KEY_NUM0; break;
                    case XK_1: k = KEY_NUM1; break;
                    case XK_2: k = KEY_NUM2; break;
                    case XK_3: k = KEY_NUM3; break;
                    case XK_4: k = KEY_NUM4; break;
                    case XK_5: k = KEY_NUM5; break;
                    case XK_6: k = KEY_NUM6; break;
                    case XK_7: k = KEY_NUM7; break;
                    case XK_8: k = KEY_NUM8; break;
                    case XK_9: k = KEY_NUM9; break;

                    case XK_Shift_L: k = KEY_LSHIFT; break;
                    case XK_Shift_R: k = KEY_RSHIFT; break;
                    case XK_Control_L: k = KEY_LCTRL; break;
                    case XK_Control_R: k = KEY_RCTRL; break;
                    case XK_Alt_L: k = KEY_LALT; break;
                    case XK_Alt_R: k = KEY_RALT; break;

                    case XK_F1: k = KEY_F1; break;
                    case XK_F2: k = KEY_F2; break;
                    case XK_F3: k = KEY_F3; break;
                    case XK_F4: k = KEY_F4; break;

                    default: break;
                }

                if (k != KEY_UNKNOWN) setKey(input, k, down);
                break;
            }

            default: break;
        }
    }

    if (input->mouse_grabbed && (input->mouse_dx != 0 || input->mouse_dy != 0))
    {
        XWarpPointer(display, None, input->grab_window,
                     0, 0, 0, 0, input->center_x, input->center_y);
        XFlush(display);
        input->last_x = input->center_x;
        input->last_y = input->center_y;
    }

    return shouldClose;
}

#endif

inline void updateInput(Input *input)
{
    for (int i = 0; i < KEY_COUNT; i++)
        input->key_prev[i] = input->key_curr[i];
    for (int i = 0; i < MOUSE_COUNT; i++)
        input->mouse_prev[i] = input->mouse_curr[i];
}

inline bool isKeyDown(const Input *input, const Key key)
{
    return key < KEY_COUNT ? input->key_curr[key] : false;
}

inline bool isKeyPressed(const Input *input, const Key key)
{
    return key < KEY_COUNT ? (input->key_curr[key] && !input->key_prev[key]) : false;
}

inline bool isKeyReleased(const Input *input, const Key key)
{
    return key < KEY_COUNT ? (!input->key_curr[key] && input->key_prev[key]) : false;
}

inline bool isMouseDown(const Input *input, const MouseButton btn)
{
    return btn < MOUSE_COUNT ? input->mouse_curr[btn] : false;
}

inline bool isMousePressed(const Input *input, const MouseButton btn)
{
    return btn < MOUSE_COUNT ? (input->mouse_curr[btn] && !input->mouse_prev[btn]) : false;
}

inline bool isMouseReleased(const Input *input, const MouseButton btn)
{
    return btn < MOUSE_COUNT ? (!input->mouse_curr[btn] && input->mouse_prev[btn]) : false;
}

inline void getMousePosition(const Input *input, int *x, int *y)
{
    if (x) *x = input->mouse_x;
    if (y) *y = input->mouse_y;
}

inline void getMouseDelta(const Input *input, int *dx, int *dy)
{
    if (dx) *dx = input->mouse_dx;
    if (dy) *dy = input->mouse_dy;
}

#ifdef SDL_IMPLEMENTATION

inline void grabMouse(SDL_Window *window, const int width, const int height, Input *input)
{
    if (input->mouse_grabbed) return;

    input->center_x = width / 2;
    input->center_y = height / 2;
    input->grab_window = (void*)window;

    SDL_SetRelativeMouseMode(true);

    input->mouse_grabbed = true;
    input->mouse_dx = 0;
    input->mouse_dy = 0;
}

inline void releaseMouse(SDL_Window *window, Input *input)
{
    (void)window;
    if (!input->mouse_grabbed) return;

    SDL_SetRelativeMouseMode(false);

    input->mouse_grabbed = false;
    input->mouse_dx = 0;
    input->mouse_dy = 0;
}

#else

inline void grabMouse(Display *display, const Window window,
                      const int width, const int height, Input *input)
{
    if (input->mouse_grabbed) return;

    input->center_x = width / 2;
    input->center_y = height / 2;
    input->grab_window = window;

    XColor dummy;
    const char data[1] = {0};
    const Pixmap blank = XCreateBitmapFromData(display, window, data, 1, 1);
    const Cursor cursor = XCreatePixmapCursor(display, blank, blank,
                                              &dummy, &dummy, 0, 0);
    XFreePixmap(display, blank);

    XDefineCursor(display, window, cursor);
    XGrabPointer(display, window, True,
                 PointerMotionMask | ButtonPressMask | ButtonReleaseMask,
                 GrabModeAsync, GrabModeAsync, window, None, CurrentTime);

    XWarpPointer(display, None, window, 0, 0, 0, 0,
                 input->center_x, input->center_y);
    XFlush(display);

    input->last_x = input->center_x;
    input->last_y = input->center_y;
    input->mouse_grabbed = true;
    input->mouse_dx = 0;
    input->mouse_dy = 0;
}

inline void releaseMouse(Display *display, const Window window, Input *input)
{
    if (!input->mouse_grabbed) return;

    XUngrabPointer(display, CurrentTime);
    XDefineCursor(display, window, None);
    input->mouse_grabbed = false;
    input->mouse_dx = 0;
    input->mouse_dy = 0;
}

#endif

inline bool isMouseGrabbed(const Input *input)
{
    return input->mouse_grabbed;
}

#endif // KEYS_IMPLEMENTATION
#endif // WRAPPER_KEYS_H

// ============================================================================
// Vector math
// ============================================================================
#ifndef WRAPPER_MATH_H
#define WRAPPER_MATH_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 origin, direction;
} Ray;

/*  -> Example:
 *  Vec3 v = vec3(1.0f, 2.0f, 3.0f);
 */
Vec3  vec3(float x, float y, float z);

/*  -> Example:
 *  Vec3 c = add(a, b);
 */
Vec3  add(Vec3 a, Vec3 b);

/*  -> Example:
 *  Vec3 c = sub(a, b);
 */
Vec3  sub(Vec3 a, Vec3 b);

/*  -> Example:
 *  Vec3 v = mul(a, 2.0f);
 */
Vec3  mul(Vec3 v, float t);

/*  -> Example:
 *  Vec3 c = vmul(a, b);
 */
Vec3  vmul(Vec3 a, Vec3 b);

/*  -> Example:
 *  Vec3 v = vdiv(a, 2.0f);
 */
Vec3  vdiv(Vec3 v, float t);

/*  -> Example:
 *  float d = dot(a, b);
 */
float dot(Vec3 a, Vec3 b);

/*  -> Example:
 *  Vec3 n = cross(a, b);
 */
Vec3  cross(Vec3 a, Vec3 b);

/*  -> Example:
 *  float length = len(v);
 */
float len(Vec3 v);

/*  -> Example:
 *  Vec3 unit = norm(v);
 */
Vec3  norm(Vec3 v);

/*  -> Example:
 *  Vec3 reflected = reflect(incident, normal);
 */
Vec3  reflect(Vec3 v, Vec3 n);

#ifdef MATH_IMPLEMENTATION

inline Vec3 vec3(const float x, const float y, const float z)
{
    return (Vec3) {x, y, z};
}

inline Vec3 add(const Vec3 a, const Vec3 b)
{
    return (Vec3) {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
    };
}

inline Vec3 sub(const Vec3 a, const Vec3 b)
{
    return (Vec3) {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    };
}

inline Vec3 mul(const Vec3 v, const float t)
{
    return (Vec3) {
        v.x * t,
        v.y * t,
        v.z * t,
    };
}

inline Vec3 vmul(const Vec3 a, const Vec3 b)
{
    return (Vec3) {
        a.x * b.x,
        a.y * b.y,
        a.z * b.z,
    };
}

inline Vec3 vdiv(const Vec3 v, const float t)
{
    const float inv = 1.0f / t;
    return (Vec3) {
        v.x * inv,
        v.y * inv,
        v.z * inv,
    };
}

inline float dot(const Vec3 a, const Vec3 b)
{
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

inline Vec3 cross(const Vec3 a, const Vec3 b)
{
    return (Vec3) {
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z),
        (a.x * b.y) - (a.y * b.x),
    };
}

inline float len(const Vec3 v)
{
    return sqrtf(dot(v,v));
}

inline Vec3 norm(const Vec3 v)
{
    const float l2 = dot(v,v);
    if (l2 > 0.0f)
    {
        const float inv = 1.0f / sqrtf(l2);
        return (Vec3) {
            v.x * inv,
            v.y * inv,
            v.z * inv,
        };
    }
    return v;
}

inline Vec3 reflect(const Vec3 v, const Vec3 n)
{
    const float d = 2.0f * dot(v,n);
    return (Vec3) {
        v.x - (d * n.x),
        v.y - (d * n.y),
        v.z - (d * n.z),
    };
}

#endif // MATH_IMPLEMENTATION
#endif // WRAPPER_MATH_H

#ifndef WRAPPER_CAMERA_H
#define WRAPPER_CAMERA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Camera {
    Vec3 position;
    float yaw;      // Rotation in degrees
    float pitch;    // Rotation in degrees
    float fov;      // Field of view in degrees
    Vec3 front;     // Forward direction vector
    Vec3 right;     // Right direction vector
    Vec3 up;        // Up direction vector
} Camera;

// Initialize camera with default position and orientation
/*  -> Example:
 *  Camera camera;
 *  cameraInit(&camera);
 *  camera.position = vec3(0.0f, 1.0f, 5.0f);
 */
void cameraInit(Camera *cam);

// Update camera vectors based on yaw/pitch (automatically clamps pitch)
/*  -> Example:
 *  camera.yaw += 10.0f;
 *  cameraUpdate(&camera);
 */
void cameraUpdate(Camera *cam);

// Move camera in given direction by speed amount
/*  -> Example:
 *  if (isKeyDown(&input, KEY_W)) cameraMove(&camera, camera.front, 0.1f);
 */
void cameraMove(Camera *cam, Vec3 direction, float speed);

// Rotate camera by delta angles in degrees
/*  -> Example:
 *  cameraRotate(&camera, mouse_dx * sensitivity, -mouse_dy * sensitivity);
 */
void cameraRotate(Camera *cam, float dyaw, float dpitch);

// Generate ray from camera with pre-scaled viewport offsets
/*  -> Example:
 *  float* u_offsets = (float*)malloc(win.width * sizeof(float));
    float* v_offsets = (float*)malloc(win.height * sizeof(float));
    assert(u_offsets && v_offsets && "Failed to allocate viewport offset buffers");

    for (int x = 0; x < win.width; x++)  u_offsets[x] = ((float)x / (float)(win.width - 1) - 0.5f) * viewport_width;
    for (int y = 0; y < win.height; y++) v_offsets[y] = ((float)(win.height - 1 - y) / (float)(win.height - 1) - 0.5f) * viewport_height;

    while (1) {
        ...
        #pragma omp parallel for schedule(dynamic) default(none) shared(win, camera, u_offsets, v_offsets)
        for (int y = 0; y < win.height; y++) {
            for (int x = 0; x < win.width; x++) {
                const Ray ray = cameraGetRay(&camera, u_offsets[x], v_offsets[y]);
                const Vec3 color = calculate_ray_color(ray, MAX_BOUNCES);
                win.buffer[y * win.width + x] = uint32(color);
        }
    }
}*/
Ray cameraGetRay(const Camera* cam, float u_scaled, float v_scaled);

#ifdef __cplusplus
}
#endif

#ifdef CAMERA_IMPLEMENTATION

inline void cameraInit(Camera *cam)
{
    cam->position = vec3(0.0f, 0.0f, 0.0f);
    cam->yaw = 0.0f;
    cam->pitch = 0.0f;
    cam->fov = 60.0f;
    cameraUpdate(cam);
}

inline void cameraUpdate(Camera *cam)
{
    // Clamp pitch to avoid gimbal lock
    if (cam->pitch > 89.0f) cam->pitch = 89.0f;
    if (cam->pitch < -89.0f) cam->pitch = -89.0f;

    const float yaw_rad = cam->yaw * M_PI / 180.0f;
    const float pitch_rad = cam->pitch * M_PI / 180.0f;

    cam->front = vec3(
        cosf(yaw_rad) * cosf(pitch_rad),
        sinf(pitch_rad),
        sinf(yaw_rad) * cosf(pitch_rad)
    );

    cam->front = norm(cam->front);
    cam->right = norm(cross(cam->front, vec3(0, 1, 0)));
    cam->up    = cross(cam->right, cam->front);
}

inline void cameraMove(Camera *cam, const Vec3 direction, const float speed)
{
    cam->position = add(cam->position, mul(direction, speed));
}

inline void cameraRotate(Camera *cam, const float dyaw, const float dpitch)
{
    cam->yaw += dyaw;
    cam->pitch += dpitch;
    cameraUpdate(cam);
}

inline Ray cameraGetRay(const Camera* cam, const float u_scaled, const float v_scaled)
{
    Vec3 rd = add(cam->front, add(mul(cam->up, v_scaled), mul(cam->right, u_scaled)));
    rd = norm(rd);
    return (Ray){cam->position, rd};
}

#endif // CAMERA_IMPLEMENTATION
#endif // WRAPPER_CAMERA_H

// ============================================================================
// 3D model with materials and transforms
// ============================================================================
#ifndef WRAPPER_MODEL_H
#define WRAPPER_MODEL_H

#define INITIAL_VERTEX_CAPACITY 1024
#define INITIAL_TRIANGLE_CAPACITY 2048

#ifdef __cplusplus
extern "C" {
#endif

// Material properties for rendering
typedef struct {
    Vec3 color;
    float reflectivity;  // 0.0 = matte, 1.0 = mirror
	float specular;
} Material;

// Triangle primitive for meshes
typedef struct {
    Vec3 v0, v1, v2;
} Triangle;

// 3D model with transform and material
typedef struct {
    Triangle* triangles;              // Original mesh triangles
    Triangle* transformed_triangles;  // World-space transformed triangles
    int num_triangles;
    int capacity;                      // Allocated triangle capacity
    Vec3 position;
    float rot_x, rot_y, rot_z;         // Euler angles in radians
    Vec3 scale;
    Material mat;
} Model;

// Create new model in storage array (returns NULL if array is full)
/*  -> Example:
 *  Model* cube = modelCreate(scene_models, &num_models, MAX_MODELS, vec3(1,0,0), 0.5f);
 */
Model* modelCreate(Model* storage, int* count, int max, Vec3 color, float refl, float spec);

// Load OBJ file into model (dynamically allocates triangles)
/*  -> Example:
 *  modelLoad(cube, "res/cube.obj");
 */
void modelLoad(Model* m, const char* path);

// Free model triangle data
/*  -> Example:
 *  modelFree(cube);
 */
void modelFree(Model* m);

// Set model transform (position, rotation in radians, scale)
/*  -> Example:
 *  modelTransform(cube, vec3(0,0,0), vec3(0, M_PI/4, 0), vec3(1,1,1));
 */
void modelTransform(Model* m, Vec3 pos, Vec3 rot, Vec3 scale);

// Apply transforms to all models in array (call after changing transforms)
/*  -> Example:
 *  modelUpdate(scene_models, num_models);
 */
void modelUpdate(const Model* models, int count);

#ifdef __cplusplus
}
#endif

#ifdef MODEL_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>

// Rotation helpers
static inline Vec3 rotate_x(const Vec3 v, const float a)
{
    const float c = cosf(a), s = sinf(a);
    return vec3(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

static inline Vec3 rotate_y(const Vec3 v, const float a)
{
    const float c = cosf(a), s = sinf(a);
    return vec3(v.x * c - v.z * s, v.y, v.x * s + v.z * c);
}

static inline Vec3 rotate_z(const Vec3 v, const float a)
{
    const float c = cosf(a), s = sinf(a);
    return vec3(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}

static inline Vec3 transform_vertex(Vec3 v, const Model* m)
{
    v = vmul(v, m->scale);
    if (m->rot_z) v = rotate_z(v, m->rot_z);
    if (m->rot_x) v = rotate_x(v, m->rot_x);
    if (m->rot_y) v = rotate_y(v, m->rot_y);
    return add(v, m->position);
}

inline Model* modelCreate(Model* storage, int* count, const int max, const Vec3 color, const float refl, const float spec)
{
    if (*count >= max) return NULL;
    Model* m = &storage[(*count)++];
    m->triangles = NULL;
    m->transformed_triangles = NULL;
    m->num_triangles = 0;
    m->capacity = 0;
    m->position = (Vec3){0, 0, 0};
    m->scale = (Vec3){1.0f, 1.0f, 1.0f};
    m->rot_x = 0; m->rot_y = 0; m->rot_z = 0;
    m->mat = (Material){color, refl, spec};
    return m;
}

inline void modelFree(Model* m)
{
    if (m->triangles)
    {
        free(m->triangles);
        m->triangles = NULL;
    }

    if (m->transformed_triangles)
    {
        free(m->transformed_triangles);
        m->transformed_triangles = NULL;
    }

    m->num_triangles = 0;
    m->capacity = 0;
}

inline void modelTransform(Model* m, const Vec3 pos, const Vec3 rot, const Vec3 scale)
{
    m->position = pos;
    m->rot_x = rot.x;
    m->rot_y = rot.y;
    m->rot_z = rot.z;
    m->scale = scale;
}

inline void modelUpdate(const Model* models, const int count)
{
    for (int i = 0; i < count; i++)
    {
        const Model* m = &models[i];
        for (int j = 0; j < m->num_triangles; j++)
        {
            m->transformed_triangles[j].v0 = transform_vertex(m->triangles[j].v0, m);
            m->transformed_triangles[j].v1 = transform_vertex(m->triangles[j].v1, m);
            m->transformed_triangles[j].v2 = transform_vertex(m->triangles[j].v2, m);
        }
    }
}

inline void modelLoad(Model* m, const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open OBJ file: %s\n", path);
        return;
    }

    // Dynamic arrays for vertices and triangles
    int vert_capacity = INITIAL_VERTEX_CAPACITY;
    int tri_capacity  = INITIAL_TRIANGLE_CAPACITY;

    Vec3* verts     = malloc(vert_capacity * sizeof(Vec3));
    Triangle* tris = malloc(tri_capacity * sizeof(Triangle));
    assert(verts && tris && "Failed to allocate OBJ parsing buffers");

    int nv = 0, nt = 0;
    char buf[256];

    while (fgets(buf, sizeof(buf), f))
    {
        if (buf[0] == 'v' && buf[1] == ' ')
        {
            // Vertex line
            if (nv >= vert_capacity)
            {
                vert_capacity *= 2;
                verts = (Vec3*)realloc(verts, vert_capacity * sizeof(Vec3));
                assert(verts && "Failed to reallocate vertex buffer");
            }

            float x, y, z;
            sscanf(buf + 2, "%f %f %f", &x, &y, &z);
            verts[nv++] = vec3(x, y, z);
        }
        else if (buf[0] == 'f')
        {
            // Face line (only supports triangulated meshes)
            if (nt >= tri_capacity)
            {
                tri_capacity *= 2;
                tris = (Triangle*)realloc(tris, tri_capacity * sizeof(Triangle));
                assert(tris && "Failed to reallocate triangle buffer");
            }

            int a, b, c;
            if (sscanf(buf + 2, "%d %d %d", &a, &b, &c) == 3)
            {
                // OBJ indices are 1-based
                if (a > 0 && a <= nv && b > 0 && b <= nv && c > 0 && c <= nv)
                {
                    tris[nt++] = (Triangle){verts[a-1], verts[b-1], verts[c-1]};
                }
            }
        }
    }
    fclose(f);

    // Allocate exact size for model
    m->triangles             = malloc(nt * sizeof(Triangle));
    m->transformed_triangles = malloc(nt * sizeof(Triangle));
    assert(m->triangles && m->transformed_triangles && "Failed to allocate model triangles");

    memcpy(m->triangles, tris, nt * sizeof(Triangle));
    m->num_triangles = nt;
    m->capacity = nt;

    // Free temporary buffers
    free(verts);
    free(tris);

    printf("Loaded %s: %d vertices, %d triangles\n", path, nv, nt);
}

#endif // MODEL_IMPLEMENTATION
#endif // WRAPPER_MODEL_H