#ifndef __JitteredSampler__
#define __JitteredSampler__

#include "Sampler.h"

#include <cuda_runtime.h>
#include "cuda_math.h"

class JitteredSampler : public Sampler {
public:
	__device__
	virtual void GenerateSamples(int num_of_samples, int num_of_sets = 1);
};

#endif