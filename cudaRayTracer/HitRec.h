#ifndef __HitRec__
#define __HitRec__

#include "World.h"
#include <cuda_runtime.h>
#include "RGBColor.h"

struct HitRec {
public:
	bool isHit;
	float3 hit_point;
	float tmin;
	RGBColor color;
};
#endif