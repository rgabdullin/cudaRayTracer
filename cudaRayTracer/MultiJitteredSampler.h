#ifndef __MultiJitteredSampler__
#define __MultiJitteredSampler__

#include "Sampler.h"

#include <cuda_runtime.h>
#include "cuda_math.h"

class MultiJitteredSampler : public Sampler {
protected:
	__device__
	void shuffle_coordinates(int offset);
public:
	__device__
	virtual void GenerateSamples(int num_of_samples, int num_of_sets = 1);
};

#endif