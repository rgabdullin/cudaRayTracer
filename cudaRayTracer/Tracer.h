#ifndef __Tracer__
#define __Tracer__

#include "World.h"

class World;

class Tracer {
protected:
	World* wr;
public:
	__device__
	Tracer(World * l_wr);
	__device__
	virtual void TraceRay(Ray* ray, HitRec* hr) = 0;
};

#endif