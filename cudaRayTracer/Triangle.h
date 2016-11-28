#ifndef __Triangle__
#define __Triangle__

#include "GObject.h"
#include "cuda_math.h"

class Triangle : public GObject
{
	float3 points[3];
	float3 normal;
	float3 color;
public:
	__host__ __device__
	Triangle();
	__host__ __device__
	void init(float3 p1, float3 p2, float3 p3, float3 color = make_float3(1, 1, 1));
	__device__
	virtual bool Intersection(const Ray& ray, HitRec& hr);
};

#endif