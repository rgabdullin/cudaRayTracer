#define SYNC_AND_CHECK_CUDA_ERRORS {cudaDeviceSynchronize(); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); return 1; }}

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//#include "UnifiedMemory.h"

class Vector3f{
	float x,y,z; 
	int* test;
public:
	__host__ __device__
	Vector3f(float l_x ,float l_y ,float l_z) :
		x(l_x),
		y(l_y),
		z(l_z)
	{
		printf("Vector3f(%f,%f,%f)\n", l_x, l_y, l_z);
		test = (int*)Vector3f::operator new(sizeof(int));
		*test = 1;
	}
	__host__ __device__
	friend Vector3f operator+(const Vector3f& a, const Vector3f& b);
	__host__ __device__
		float GetX() { return x; }
	__host__ __device__
		float GetY() { return y; }
	__host__ __device__
		float GetZ() { return z; }

	__host__ __device__
	void *operator new(size_t len) {
		printf("Allocating Unified Memory: %u bytes\n", len);
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}
	__host__ __device__
	void operator delete(void *ptr) {
		printf("Deallocating Unified Memory\n");
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};
__host__ __device__
Vector3f operator+(const Vector3f& a, const Vector3f& b) {
	return Vector3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ 
void test_gpu(Vector3f* a) {
	printf("GPU Test\n");

	Vector3f b(-1, -1, -1);

	Vector3f c(2, 3, 4);

	*a = b + c;

	printf("GPU Ok\n");
}

int main(void) {
	Vector3f* A = new Vector3f(0, 0, 0);

	test_gpu <<<1, 1>>> (A);
	SYNC_AND_CHECK_CUDA_ERRORS;

	printf("A = (%f,%f,%f)\n", A->GetX(), A->GetY(), A->GetZ());

	delete A;
	system("pause");
}