#ifndef __Plane__
#define __Plane__

#include "GObject.h"
#include "cuda_math.h"

class Plane : public GObject
{
	float3 point;
	float3 normal;
	float3 color;
public:
	__host__ __device__
	Plane();
	__host__ __device__
	void init(float3 p, float3 n, float3 color = make_float3(0.3, 0.3, 0.3));
	__device__
	virtual bool Intersection(const Ray& ray, HitRec& hr);
};

#endif