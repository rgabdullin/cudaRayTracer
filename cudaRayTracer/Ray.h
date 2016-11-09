#ifndef __Ray__
#define __Ray__

#include <cuda_runtime.h>

typedef struct _Ray {
	float3 origin;
	float3 direction;
	int2 pixel;
} Ray;
#endif