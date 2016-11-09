#pragma once

#include <cuda.h>
#include <device_launch_parameters.h>

class Object {
public:
//	__device__ __host__ virtual float Intersection(float3 pos, float3 dir, float3& hPos, float3& hNor) = 0;
//	__device__ __host__ virtual float3 getObjectColor(float3 pos) = 0;
//	__device__ __host__ virtual float3 getNormal(float3 pos) = 0;
	__device__ __host__ virtual void print() = 0;
};

typedef struct __Sphere {
	float3 pos;
	float Radius;
} __Sphere;

class Sphere : public Object {	
public:
	__Sphere data;
	__device__ __host__ Sphere(__Sphere data);
	//	__device__ __host__ virtual float Intersection(float3 pos, float3 dir, float3& hPos, float3& hNor);
	//	__device__ __host__ virtual float3 getObjectColor(float3 pos);
	//	__device__ __host__ virtual float3 getNormal(float3 pos);
	__device__ __host__ virtual void print(void);
};

typedef struct __Triangle {
	float3 p[3];
} __Triangle;

class Triangle : public Object {
public:
	__Triangle data;
	__device__ __host__ Triangle(__Triangle data);
	//	__device__ __host__ virtual float Intersection(float3 pos, float3 dir, float3& hPos, float3& hNor);
	//	__device__ __host__ virtual float3 getObjectColor(float3 pos);
	//	__device__ __host__ virtual float3 getNormal(float3 pos);
	__device__ __host__ virtual void print(void);
};
namespace test {
	__global__ void GPUCreateTestObjectArray(Object** dev_array);
	__global__ void GPUDestroyTestObjectArray(Object** dev_array);
	__global__ void kernel(Object ** ptr);
};