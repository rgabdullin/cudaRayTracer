#define SYNC_AND_CHECK_CUDA_ERRORS {cudaStreamSynchronize(0); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); exit(1); }}

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 8
#define FRAMESIZE_X 1024
#define FRAMESIZE_Y 1024

#define NO_PRINTF

#include "cuda_math.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "PrimaryRayTracer.h"
#include "World.h"

/* prototypes */
__global__ void ClearGPU(World* ptr);
__global__ void GenPrimaryRays_Frame(const World& w, Ray* rays_ptr, int2 leftbottom);
__global__ void TracePrimaryRays_Frame(World & wr, Ray * rays_ptr);
__global__ void build_gpu(World * w);
__host__ __device__ void make_HitRec(HitRec* hr);

/* Tracer */
__device__
Tracer::Tracer(World* l_wr) :
	wr(l_wr)
{}

/* PrimaryRayTracer */
__device__
PrimaryRayTracer::PrimaryRayTracer(World* l_wr) :
	Tracer(l_wr)
{}

__device__
void PrimaryRayTracer::TraceRay(Ray* ray, HitRec* hr) {
	make_HitRec(hr);

	wr->isHitSceneObject(ray, hr);
}

/* Sphere */
__host__ __device__
Sphere::Sphere() :
	origin(make_float3(0, 0, 0)),
	color(make_float3(1, 1, 1)),
	radius(1)
{}

__host__ __device__
void Sphere::init(float3 l_origin, float l_radius, float3 l_color) {
	origin = l_origin;
	radius = l_radius;
	color = l_color;
}

__device__
bool Sphere::Intersection(const Ray& ray, HitRec& hr) {
	float3 v = ray.origin - origin;
	float3 origin = ray.origin;
	float3 dir = ray.direction;

	float A = dot(dir, dir);
	float B = 2 * dot(v, dir);
	float C = dot(v, v) - radius * radius;

	float D = B * B - 4 * A * C;
	if (D >= 0) {
		float t1 = (-B - sqrt(D)) / A * 0.5;
		float t2 = (-B + sqrt(D)) / A * 0.5;
		float t = -1;
		if (t1 > eps)
			t = t1;
		else
			if (t2 > eps)
				t = t2;
		if (t > 0 && (!hr.isHit || t < hr.tmin)) {
			hr.isHit = true;
			hr.tmin = t;
			hr.hit_point = origin + dir * t;
			hr.color = this->color;

			return true;
		}
	}
	return false;
}

/* Triangle */


/* HitRec */
__host__ __device__
void make_HitRec(HitRec* hr) {
	hr->isHit = false;
	hr->hit_point = make_float3(0.0f, 0.0f, 0.0f);
	hr->tmin = 1e8;
	hr->color = make_float3(0.0f, 0.0f, 0.0f);
}

/* ViewPlane */
void ViewPlane::init(float3 l_origin, float3 l_direction, int2 l_res, float l_height) {
	origin = l_origin;
	direction = normalize(l_direction);
	res = l_res;
	height = l_height;
	psize = l_height / (res.y + 1);
}

/* World */
World::World(void) {}

void World::init(int2 res, float size) {
	vp.init(make_float3(0, 0, 6), make_float3(0, 0, -1), res, size);

	background_color = make_float3(0, 0, 0.25);

	cudaMallocManaged(&image, sizeof(float3) * res.x * res.y);
	printf("Allocated memory to image: %.2fKBs\n", float(sizeof(float3) * res.x * res.y) / 1024);
	SYNC_AND_CHECK_CUDA_ERRORS;

	init_gpu <<< 1, 1 >>>(this);
	SYNC_AND_CHECK_CUDA_ERRORS;
}

void World::render_scene(void) {
	int width = vp.res.x;
	int height = vp.res.y;

	dim3 blockD(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 frameD(FRAMESIZE_X, FRAMESIZE_Y);
	dim3 gridD(frameD.x / blockD.x, frameD.y / blockD.y);

	dim3 framesGridD = dim3(width / frameD.x + (width % frameD.x ? 1 : 0), height / frameD.y + (height % frameD.y ? 1 : 0));

	printf("\nFrames Grid     : %d x %d\n", framesGridD.x, framesGridD.y);
	printf("Frames Dimention: %d x %d\n", frameD.x, frameD.y);
	printf("Image resolution: %d x %d\n", width, height);
	int num_of_frames = framesGridD.x * framesGridD.y;

	Ray* rays_ptr;
	cudaMalloc(&rays_ptr, sizeof(Ray) * frameD.x * frameD.y);
	printf("Allocated memory for rays_ptr: %.2f KBs\n\t", float(frameD.x * frameD.y * sizeof(Ray)) / 1024);
	SYNC_AND_CHECK_CUDA_ERRORS;

	printf("Rendering:\n");
	for (int k = 0; k < framesGridD.y; ++k)
		for (int i = 0; i < framesGridD.x; ++i) {
			#ifdef NO_PRINTF
				printf("\r\t%.2f%%", 100 * float(i + framesGridD.x * k) / num_of_frames);
			#else
				printf("Frame: (%d , %d)\n\t", i, k);
				printf("Generating Rays:");
			#endif

				GenPrimaryRays_Frame <<< gridD, blockD >>> (*this, rays_ptr, make_int2(i * frameD.x, k * frameD.y));
				SYNC_AND_CHECK_CUDA_ERRORS;

			#ifndef NO_PRINTF
				printf("OK\n\t");
			#endif

			#ifndef NO_PRINTF
				printf("Tracing Rays:");
			#endif

				TracePrimaryRays_Frame <<< gridD, blockD >>> (*this, rays_ptr);
				SYNC_AND_CHECK_CUDA_ERRORS;

			#ifndef NO_PRINTF
				printf("OK\n");
			#endif
		}
	#ifndef NO_PRINTF
		printf("Deallocating Memory\n\t");
	#endif

		cudaFree(rays_ptr);
		printf("\r\tOK           \n");
}

void World::save_image(std::string filename) {
	fipImage img(FIT_BITMAP, vp.res.x, vp.res.y, 24);
	BYTE* ptr = img.accessPixels();
	int pitch = img.getScanWidth();
	int bmask = FreeImage_GetBlueMask(img);
	int rmask = FreeImage_GetRedMask(img);
	int r = 0, b = 2;
	if (rmask > bmask) {
		r = 2; b = 0;
	}
	float3 a0 = make_float3(0, 0, 0);
	float3 a1 = make_float3(1, 1, 1);
	for(int k = 0; k < vp.res.y; ++k)
		for (int i = 0; i < vp.res.x; ++i) {
			float3 v = clamp(image[i + k * vp.res.x], a0, a1) * 255.0f;
			ptr[pitch * k + 3 * i + r] = unsigned char(v.x);
			ptr[pitch * k + 3 * i + 1] = unsigned char(v.y);
			ptr[pitch * k + 3 * i + b] = unsigned char(v.z);
		}
	img.save(filename.c_str());
	img.clear();
}

void World::clear() {
	cudaFree(image);

	ClearGPU << < 1, num_of_objects >> > (this);
	SYNC_AND_CHECK_CUDA_ERRORS;

	cudaFree(this);
}

void World::build_scene() {
	num_of_objects = 5;

	cudaMalloc(&scene_objs, sizeof(GObject*) * num_of_objects);

	build_gpu <<< 1, 1 >>> (this);

	SYNC_AND_CHECK_CUDA_ERRORS;
}

__device__ bool World::isHitSceneObject(Ray * ray, HitRec * hr) {
	for (int i = 0; i < num_of_objects; ++i)
		scene_objs[i]->Intersection(*ray, *hr);

	return hr->isHit;
}

__device__ void World::init_ray_tracers() {
	primary_ray_tracer = new PrimaryRayTracer(this);
}
/* Kernels */
__global__
void ClearGPU(World* ptr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx == 0)
		delete ptr->primary_ray_tracer;
	delete ptr->scene_objs[idx];
}

__global__
void GenPrimaryRays_Frame(const World& w, Ray* rays_ptr, int2 leftbottom) {
	int x, y, idx;
	int b_idx = threadIdx.x + threadIdx.y * blockDim.y;
	__shared__ int width; if (b_idx == 0) width = w.vp.res.x;
	__shared__ int height; if (b_idx == 0)  height = w.vp.res.y;
	__shared__ float3 direction; if (b_idx == 0)  direction = w.vp.direction;
	__shared__ float3 origin; if (b_idx == 0)  origin = w.vp.origin;
	__shared__ float psize; if (b_idx == 0)  psize = w.vp.psize;
	__syncthreads();

	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	idx = x + y * blockDim.x * gridDim.x;

	Ray ray;
	ray.origin = origin + make_float3(float(leftbottom.x + x) * psize - 0.5 * psize * width, float(leftbottom.y + y) * psize - 0.5 * psize * height, 0);
	ray.direction = direction;
	ray.pixel = make_int2(leftbottom.x + x, leftbottom.y + y);

	rays_ptr[idx] = ray;
}

__global__
void TracePrimaryRays_Frame(World & wr, Ray * rays_ptr) {
	int b_idx = blockDim.x * threadIdx.y + threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y * blockDim.x * gridDim.x;
	__shared__ int2 res; if (b_idx == 0) res = wr.vp.res;
	__syncthreads();

	if (rays_ptr[idx].pixel.x < res.x && rays_ptr[idx].pixel.y < res.y) {

		__shared__ HitRec Hrs[BLOCKSIZE_X * BLOCKSIZE_Y]; __syncthreads();

		wr.primary_ray_tracer->TraceRay(&rays_ptr[idx], &Hrs[b_idx]);

		int img_idx = rays_ptr[idx].pixel.x + res.x * rays_ptr[idx].pixel.y;
		if (Hrs[b_idx].isHit)
			wr.image[img_idx] = Hrs[b_idx].color;
		else
			wr.image[img_idx] = wr.background_color;
	}
}

__global__
void build_gpu(World * w) {
	Sphere * ptr1 = new Sphere();
	ptr1->init(make_float3(0, 0, 0), 2, make_float3(1, 0, 0));
	w->scene_objs[0] = ptr1;

	Sphere * ptr2 = new Sphere();
	ptr2->init(make_float3(2, 0.5, -5), 1, make_float3(1, 1, 0));
	w->scene_objs[1] = ptr2;

	Sphere * ptr3 = new Sphere();
	ptr3->init(make_float3(-3, 0, 2), 1.5, make_float3(0, 0, 1));
	w->scene_objs[2] = ptr3;

	Sphere * ptr4 = new Sphere();
	ptr4->init(make_float3(2, 0, 0), 1, make_float3(0, 0.3, 0));
	w->scene_objs[3] = ptr4;

	Sphere * ptr5 = new Sphere();
	ptr5->init(make_float3(0, 0, 3), 0.2, make_float3(0, 0, 0));
	w->scene_objs[4] = ptr5;

	printf("Allocated memory to scene: %.2f KBs\n", float(sizeof(Sphere) * w->num_of_objects) / 1024);
}

__global__
void init_gpu(World * w) {
	w->init_ray_tracers();
}