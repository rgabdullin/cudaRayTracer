#ifndef __ViewPlane__
#define __ViewPlane__

#include <cuda_runtime.h>
#include <vector_types.h>

class ViewPlane {
public:
	float3 origin;
	float psize;
	float3 direction;
	float height;
	int2 res;

	void init(float3 l_origin, float3 l_direction, int2 l_res, float l_height);
};

#endif