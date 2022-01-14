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
R_VISION_API void* c_vision_ops_nms (void* dets, void* scores, double iou_threshold);


