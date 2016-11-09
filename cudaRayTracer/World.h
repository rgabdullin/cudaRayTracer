#ifndef __World__
#define __World__

#include <cuda_runtime.h>
#include <string>
#include "FreeImagePlus.h"

#include "RGBColor.h"
#include "Sphere.h"
#include "ViewPlane.h"
#include "Tracer.h"

class Tracer;

class World {
public:
//FIELDS
	ViewPlane vp;
	float3* image;

	RGBColor background_color;
	int num_of_objects;
	GObject ** scene_objs;

	Tracer * primary_ray_tracer;
//FUNCTIONS
	World(void);

	void init(int2 res = make_int2(1024, 512), float size = 1);
	void build_scene();
	void render_scene(void);
	void save_image(std::string filename);
	void clear(void);
	__device__ bool isHitSceneObject(Ray* ray, HitRec* hr);
	__device__ void init_ray_tracers();

	__global__ friend void GenPrimaryRays_Frame(const World&, Ray*, int2);
	__global__ friend void build_gpu(World *);
	__global__ friend void init_gpu(World *);
};
#endif