#include "Objects.h"

#include "cuda_math/cuda_math.h"
#include <cstdio>

Sphere::Sphere(__Sphere data)
{
	this->data = data;
}

__device__ __host__ void Sphere::print(void)
{
	printf("This is sphere: {(%.2f,%.2f,%.2f), %.2f}\n",
		data.pos.x,
		data.pos.y,
		data.pos.z,
		data.Radius
	);
}

Triangle::Triangle(__Triangle data)
{
	this->data = data;
}

__device__ __host__ void Triangle::print(void)
{
	printf("This is triangle: {(%.2f,%.2f,%.2f), (%.2f,%.2f,%.2f), (%.2f,%.2f,%.2f)}\n",
		data.p[0].x,
		data.p[0].y,
		data.p[0].z,
		data.p[1].x,
		data.p[1].y,
		data.p[1].z,
		data.p[2].x,
		data.p[2].y,
		data.p[2].z
		);
}

__global__ void test::GPUCreateTestObjectArray(Object** dev_array) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx % 4) {
		float3 p1 = make_float3(0, 0, idx);
		float3 p2 = make_float3(idx + 1, 0, 0);
		float3 p3 = make_float3(0, idx + 1, 0);
		__Triangle tr;
		tr.p[0] = p1; tr.p[1] = p2; tr.p[2] = p3;
		dev_array[idx] = new Triangle(tr);
	}
	else {
		float3 p = make_float3(0, 0, idx);
		float radius = idx;
		__Sphere sph;
		sph.pos = p;
		sph.Radius = radius;
		dev_array[idx] = new Sphere(sph);
	}

	__syncthreads();
}
__global__ void test::GPUDestroyTestObjectArray(Object** dev_array) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	delete dev_array[idx];

	__syncthreads();
}
__global__ void test::kernel(Object** ptr) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	ptr[idx]->print(); printf("%d\n", idx);

	__syncthreads();
}