#ifndef __Sampler__
#define __Sampler__

#define SAMPLER_NUM_OF_SETS 101

#include <cuda_runtime.h>
#include "cuda_math.h"

class Sampler {
protected:
	float2* square_samples;
	float3* hemisphere_samples;
public:
	int num_samples;
	int num_sets;

	__device__
	void init(int num_of_samples, int num_of_sets = 1, float e = 0.0f);

	__device__
	virtual void GenerateSamples(int num_of_samples, int num_of_sets = 1) = 0;

	__device__
	void MapSamplesToHemisphere(float e = 0);

	__device__
	float2 SampleUnitSquare(int num_of_sample, int num_of_set = 0);

	__device__
	float3 SampleHemisphere(int num_of_sample, int num_of_set = 0);

	__device__
	~Sampler();
};

#endif