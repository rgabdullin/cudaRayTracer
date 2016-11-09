#ifndef __GObject__
#define __GObject__

#include "Ray.h"
#include "HitRec.h"

#include <cuda.h>

class GObject
{
public:
	__device__
	virtual bool Intersection(const Ray&, HitRec&) = 0;
};

#endif