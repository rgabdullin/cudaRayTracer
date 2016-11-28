#ifndef __ViewPlane__
#define __ViewPlane__

#include <cuda_runtime.h>
#include <vector_types.h>

class ViewPlane {
public:
	float psize;
	int2 res;

	void init(int2 l_res, float l_height);
};

#endif