#ifndef __PrimaryRayTracer__
#define __PrimaryRayTracer__

#include "Tracer.h"

class PrimaryRayTracer : public Tracer {
public:
	__device__
	PrimaryRayTracer(World * l_wr);
	__device__
	virtual void TraceRay(Ray* ray, HitRec* hr);
};

#endif