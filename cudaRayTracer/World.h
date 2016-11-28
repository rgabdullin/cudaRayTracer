#ifndef __World__
#define __World__

#include <cuda_runtime.h>
#include <string>

#include "Camera.h"

#include "Samplers.h"

#include "RGBColor.h"

#include "Sphere.h"
#include "Triangle.h"
#include "Plane.h"

#include "ViewPlane.h"
#include "Tracer.h"

class Tracer;

class World {
//FIELDS
public:
	float3 *image;
	float3 *gpu_buffer;

	int num_rays_per_pixel;
	RGBColor background_color;
	int num_of_objects;
	
	Camera *camera;
	Tracer *ray_tracer;
	Sampler *pixel_sampler;
	GObject **scene_objs;
	ViewPlane vp;

	__device__ void init_samplers();
	__device__ void init_ray_tracers();

//FUNCTIONS
	World(void);

	void init(int2 res = make_int2(1024, 512), float size = 1, int num_rays_per_pixel = 1);
	void build_scene();
	void render_scene(int pixels_in_frame = 512 * 512);
	void save_image(std::string filename);
	void clear(void);

	__device__ bool isHitSceneObject(Ray* ray, HitRec* hr);

	__global__ friend void Render_Frame(const World* w, int offset);
	__global__ friend void build_gpu(World *);
	__global__ friend void init_gpu(World *);
	__global__ friend void ClearGPU(World* ptr);
};
#endif