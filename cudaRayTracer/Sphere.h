#ifndef __Sphere__
#define __Sphere__

#include "GObject.h"
#include "cuda_math.h"

class Sphere : public GObject
{
	float3 origin;
	float3 color;
	float radius;
public:
	__host__ __device__
	Sphere();
	__host__ __device__
	void init(float3 origin, float radius, float3 color);
	__device__
	virtual bool Intersection(const Ray& ray, HitRec& hr);
};

#endif