#ifndef __Ray__
#define __Ray__

#include <cuda_runtime.h>

typedef struct _Ray {
	float3 origin;
	float3 direction;
	int image_idx;
	int sampler_point_number;
	int sampler_set_number;
} Ray;
#endif