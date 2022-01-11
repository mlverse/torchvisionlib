#ifdef _WIN32
#ifndef VISION_HEADERS_ONLY
#define R_VISION_API extern "C" __declspec(dllexport)
#else
#define R_VISION_API extern "C" __declspec(dllimport)
#endif
#else
#define R_VISION_API extern "C"
#endif

R_VISION_API int test (void* path);


