#ifndef SYNC_AND_CHECK_CUDA_ERRORS
#define SYNC_AND_CHECK_CUDA_ERRORS {cudaStreamSynchronize(0); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); fclose(stdout); exit(1); }}
#endif

#define BLOCKSIZE 128

#define NO_PRINTF

#include "FreeImagePlus.h"

#include "cuda_math.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Sampler.h"
#include "RegularSampler.h"
#include "JitteredSampler.h"
#include "MultiJitteredSampler.h"

#include "PrimaryRayTracer.h"
#include "World.h"

#include "Camera.h"
#include "OrthographicCamera.h"
#include "PinholeCamera.h"

__device__ curandState_t curand_state;

/* prototypes */
__global__ void ClearGPU(World* ptr);
__global__ void Render_Frame(const World* w, int offset);
__global__ void build_gpu(World * w);
__global__ void init_gpu(World * w);
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
	float A = dot(ray.direction, ray.direction);
	float B = 2.0f * dot(ray.origin - origin, ray.direction);
	float C = dot(ray.origin - origin, ray.origin - origin) - radius * radius;

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
			hr.hit_point = ray.origin + ray.direction * t;
			hr.color = this->color;

			return true;
		}
	}
	return false;
}

/* Triangle */

__host__ __device__
Triangle::Triangle(){}

__host__ __device__
void Triangle::init(float3 p1, float3 p2, float3 p3, float3 l_color) {
	points[0] = p1;
	points[1] = p2;
	points[2] = p3;
	normal = normalize(cross(p1 - p2, p3 - p2));
	color = l_color;
}

__device__
bool Triangle::Intersection(const Ray& ray, HitRec& hr) {
	float t = dot(points[1] - ray.origin, normal) / dot(ray.direction, normal);
	float3 hp = ray.origin + t * ray.direction;

	if (t > 0 && (dot(cross(points[1] - points[0], hp - points[0]), cross(points[2] - points[1], hp - points[1])) >= 0
		&& dot(cross(points[2] - points[1], hp - points[1]), cross(points[0] - points[2], hp - points[2])) >= 0) && (t < hr.tmin || !hr.isHit)) {
		hr.isHit = true;
		hr.tmin = t;
		hr.hit_point = hp;
		hr.color = color;
	}
	return false;
}

/* Plane */

__host__ __device__
Plane::Plane() {}

__host__ __device__
void Plane::init(float3 p, float3 n, float3 l_color) {
	point = p;
	normal = normalize(n);
	color = l_color;
}

__device__
bool Plane::Intersection(const Ray& ray, HitRec& hr) {
	float t = dot(point - ray.origin, normal) / dot(ray.direction, normal);
	if (t > 0 && (t < hr.tmin || !hr.isHit)) {
		hr.isHit = true;
		hr.tmin = t;
		hr.hit_point = ray.origin + t * ray.direction;
		hr.color = color;
	}
	return false;
}

/* HitRec */
__host__ __device__
void make_HitRec(HitRec* hr) {
	hr->isHit = false;
	hr->hit_point = make_float3(0.0f, 0.0f, 0.0f);
	hr->tmin = 1e8;
	hr->color = make_float3(0.0f, 0.0f, 0.0f);
}

/* ViewPlane */
void ViewPlane::init(int2 l_res, float l_height) {
	res = l_res;
	psize = l_height / (res.y + 1);
}

/* Camera */

__device__
Camera::Camera(World* l_wr) {
	wr = l_wr; 
}

__device__
void Camera::init(float3 l_eye, float3 l_lookat, float3 l_up) {
	eye = l_eye;
	lookat = l_lookat;
	up = l_up;
	w = normalize(eye - lookat);
	u = normalize(cross(up, w));
	v = cross(w, u);
}

/* Orthographic Camera */

__device__
OrthographicCamera::OrthographicCamera(World* l_wr) : Camera(l_wr) {}

__device__
void OrthographicCamera::MakeRay(Ray* ray, int2 pixel, float2 sample) {
	ray->origin = eye + make_float3((float(pixel.x) + sample.x) * wr->vp.psize - 0.5 * wr->vp.psize * wr->vp.res.x, (float(pixel.y) + sample.y)* wr->vp.psize - 0.5 * wr->vp.psize * wr->vp.res.y, 0);
	ray->direction = -w;
	ray->image_idx = pixel.x + pixel.y * wr->vp.res.x;
}

/* Pinhole Camera */

__device__
PinholeCamera::PinholeCamera(World* l_wr) : Camera(l_wr) {}

__device__ 
void PinholeCamera::init_pinhole(float l_dist_to_vp, float l_zoom) {
	dist_to_vp = l_dist_to_vp;
	zoom = l_zoom;
}

__device__
void PinholeCamera::MakeRay(Ray* ray, int2 pixel, float2 sample) {
	ray->origin = eye;
	float2 pixel_point = make_float2((float(pixel.x) + sample.x) * wr->vp.psize - 0.5 * wr->vp.psize * wr->vp.res.x, (float(pixel.y) + sample.y)* wr->vp.psize - 0.5 * wr->vp.psize * wr->vp.res.y);
	ray->direction = -dist_to_vp * w + pixel_point.x * u + pixel_point.y * v;
	ray->image_idx = pixel.x + pixel.y * wr->vp.res.x;
}

/* World */
World::World(void) {}

void World::init(int2 res, float size, int num_rays_per_pixel) {
	vp.init(res, size);

	this->num_rays_per_pixel = num_rays_per_pixel;

	background_color = make_float3(0, 0, 0.25);

	image = (float3*)malloc(sizeof(float3) * res.x * res.y);
	printf("Allocated memory to image: %.2fKBs\n", float(sizeof(float3) * res.x * res.y) / 1024); fflush(stdout);
	SYNC_AND_CHECK_CUDA_ERRORS;

	init_gpu <<< 1, 1 >>>(this);
	SYNC_AND_CHECK_CUDA_ERRORS;
}

void World::save_image(std::string filename) {
	printf("Saving image in file \"%s\"\n\t", filename.c_str()); fflush(stdout);
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
	printf("OK\n"); fflush(stdout);
}

void World::clear() {
	free(image);

	ClearGPU <<< 1, num_of_objects >>> (this);
	SYNC_AND_CHECK_CUDA_ERRORS;
}

void World::build_scene() {
	num_of_objects = 7;

	cudaMalloc(&scene_objs, sizeof(GObject*) * num_of_objects);

	build_gpu <<< 1, 1 >>> (this);

	SYNC_AND_CHECK_CUDA_ERRORS;
}

__device__
bool World::isHitSceneObject(Ray * ray, HitRec * hr) {
	for (int i = 0; i < num_of_objects; ++i)
		scene_objs[i]->Intersection(*ray, *hr);

	return hr->isHit;
}

__device__ 
void World::init_samplers() {
	pixel_sampler = new RegularSampler();
	
	pixel_sampler->init(num_rays_per_pixel, 67);
}
__device__ void World::init_ray_tracers() {
	ray_tracer = new PrimaryRayTracer(this);
}

void World::render_scene(int pixels_in_frame) {
	printf("Rendering scene:\n\t");
	printf("Resolution: %d x %d\n\t", vp.res.x, vp.res.y);
	printf("Sampling: %d\n\t", num_rays_per_pixel);
	printf("Frame size: %d\n\t", pixels_in_frame); fflush(stdout);
	
	cudaMalloc(&gpu_buffer, sizeof(float3) * pixels_in_frame);

	int block_size = BLOCKSIZE;
	int num_pixels = vp.res.x * vp.res.y;
	int num_frames = num_pixels / pixels_in_frame + (num_pixels % pixels_in_frame ? 1 : 0);

	printf("Frame number: %d\n", num_frames); fflush(stdout);

	int pixels_rendered = 0;
	for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
		int pixels_to_render = min(pixels_in_frame, num_pixels - pixels_rendered);
		int num_blocks = pixels_to_render / block_size + (pixels_to_render % block_size ? 1 : 0);

		Render_Frame <<< num_blocks, block_size >>> (this, pixels_rendered);
		SYNC_AND_CHECK_CUDA_ERRORS;

		cudaMemcpy(image + pixels_rendered, gpu_buffer, sizeof(float3) * pixels_to_render, cudaMemcpyDeviceToHost);
		SYNC_AND_CHECK_CUDA_ERRORS;

		pixels_rendered += pixels_to_render;
	}
}

/* Samplers */

__device__
void Sampler::init(int num_of_samples, int num_of_sets, float e) {
	GenerateSamples(num_of_samples, num_of_sets);

	MapSamplesToHemisphere(e);
}

__device__
void RegularSampler::GenerateSamples(int l_num_of_samples, int l_num_of_sets) {
	square_samples = (float2*)malloc(sizeof(float2) * l_num_of_samples * l_num_of_sets);
	int n = sqrt(float(l_num_of_samples));
	num_samples = n*n;
	num_sets = l_num_of_sets;
	for (int i = 0; i < l_num_of_sets; ++i)
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x)
				square_samples[i * num_samples + n * y + x] = make_float2(float(x) / n + 1.0 / (2 * n), float(y) / n + 1.0 / (2 * n));
}

__device__
void JitteredSampler::GenerateSamples(int l_num_of_samples, int l_num_of_sets) {

	square_samples = (float2*)malloc(sizeof(float2) * l_num_of_samples * l_num_of_sets);

	int n = sqrt(float(l_num_of_samples));
	num_samples = n*n;
	num_sets = l_num_of_sets;

	for (int i = 0; i < l_num_of_sets; ++i)
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x)
				square_samples[i * num_samples + n * y + x] = make_float2(float(x) / n + curand_uniform(&curand_state) / n, float(y) / n + curand_uniform(&curand_state) / n);
}


__device__
void MultiJitteredSampler::shuffle_coordinates(int offset) {
	for (int i = 0; i < num_samples; ++i) {
		int k = curand(&curand_state) % num_samples;
		int j = curand(&curand_state) % num_samples;
		float c;
		c = square_samples[i + offset].x;
		square_samples[i + offset].x = square_samples[k + offset].x;
		square_samples[k + offset].x = c;
		c = square_samples[i + offset].y;
		square_samples[i + offset].y = square_samples[j + offset].y;
		square_samples[j + offset].y = c;
	}
}
__device__
void MultiJitteredSampler::GenerateSamples(int l_num_of_samples, int l_num_of_sets) {
	square_samples = (float2*)malloc(sizeof(float2) * l_num_of_samples * l_num_of_sets);

	int n = sqrt(float(l_num_of_samples));
	num_samples = n*n;
	num_sets = l_num_of_sets;

	for (int i = 0; i < l_num_of_sets; ++i) {
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x)
				square_samples[i * num_samples + n * y + x] = make_float2(
					float(x) / n + (float(y) / n + curand_uniform(&curand_state) / n) / n,
					float(y) / n + (float(x) / n + curand_uniform(&curand_state) / n) / n);
		shuffle_coordinates(i * num_samples);
	}
}

__device__
void Sampler::MapSamplesToHemisphere(float e) {
	hemisphere_samples = (float3*)malloc(sizeof(float3) * num_samples * num_sets);

	for (int i = 0; i < num_samples * num_sets; ++i) {
		float3 tmp;
		float sin_phi = sin(2.0f * acos(-1.0f) * square_samples[i].x);
		float cos_phi = cos(2.0f * acos(-1.0f) * square_samples[i].x);
		float cos_theta = pow((1.0f - square_samples[i].y), 1.0f / (e + 1.0f));
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
		tmp = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
		hemisphere_samples[i] = tmp;
	}
}

__device__
float2 Sampler::SampleUnitSquare(int l_num_of_sample, int l_num_of_set) {
	return square_samples[num_samples * (l_num_of_set % num_sets) + (l_num_of_sample % num_samples)];
}

__device__
float3 Sampler::SampleHemisphere(int l_num_of_sample, int l_num_of_set) {
	return hemisphere_samples[num_samples * (l_num_of_set % num_sets) + (l_num_of_sample % num_samples)];
}

__device__
Sampler::~Sampler(void) {
	free(square_samples);
	free(hemisphere_samples);
}

/* Kernels */
__global__
void ClearGPU(World* ptr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	delete ptr->scene_objs[idx];

	if (idx == 0) {
		delete ptr->camera;
		delete ptr->ray_tracer;
		delete ptr->pixel_sampler;
	}
}

__global__ void Render_Frame(const World * w, int offset)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int b_idx = threadIdx.x;

	__shared__ int width; if (b_idx == 0) width = w->vp.res.x;
	__shared__ int height; if (b_idx == 0) height = w->vp.res.y;
	__shared__ float inv_sampler_points_num; if (b_idx == 0) inv_sampler_points_num = 1.0 / w->num_rays_per_pixel;
	__syncthreads();
	if (idx + offset < width * height) {
		int x = (idx + offset) % width;
		int y = (idx + offset) / width;

		Ray ray;
		HitRec hr;
		float2 pt;
		float3 color;

		int sampler_set_idx = curand(&curand_state) % w->pixel_sampler->num_sets;

		w->gpu_buffer[idx] = make_float3(0, 0, 0);

		for (int i = 0; i < w->num_rays_per_pixel; ++i){
			ray.sampler_point_number = i;
			ray.sampler_set_number = sampler_set_idx;

			pt = w->pixel_sampler->SampleUnitSquare(ray.sampler_point_number, ray.sampler_set_number);

			w->camera->MakeRay(&ray, make_int2(x, y), pt);

			make_HitRec(&hr);

			w->ray_tracer->TraceRay(&ray, &hr);

			color = w->background_color;

			if (hr.isHit)
				color = hr.color;

			w->gpu_buffer[idx] += color * inv_sampler_points_num;
			__syncthreads();
		}
	}
}

__global__
void build_gpu(World * w) {
	PinholeCamera *c = new PinholeCamera(w);
	c->init(make_float3(0, 0, 10), make_float3(0, 0, 0), make_float3(0, 1, 0));
	c->init_pinhole(4);
	w->camera = c;

	Sphere * ptr1 = new Sphere();
	ptr1->init(make_float3(0, 0, 0), 2, make_float3(1, 0, 0));
	w->scene_objs[0] = ptr1;

	Sphere * ptr2 = new Sphere();
	ptr2->init(make_float3(2, 0.5, -5), 3, make_float3(1, 1, 0));
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

	Plane * ptr6 = new Plane();
	ptr6->init(make_float3(0, -5, 0), make_float3(0, 1, 0));
	w->scene_objs[5] = ptr6;

	Triangle * ptr7 = new Triangle();
	ptr7->init(make_float3(0.5, 0, 5), make_float3(0, 0, 3), make_float3(0, 0.5, 3));
	w->scene_objs[6] = ptr7;
}

__global__
void init_gpu(World * w) {
	curand_init(23041996, 0, 0, &curand_state);
	
	w->init_samplers();
	w->init_ray_tracers();
}