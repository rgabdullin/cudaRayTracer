#include "Sampler.h"
#include "RegularSampler.h"
#include "JitteredSampler.h"
#include "MultiJitteredSampler.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_math.h"
#include "curand.h"
#include "curand_kernel.h"

#include <cstdlib>
#include <cstdio>

__device__ curandState_t curand_state;

__device__
void Sampler::init(int num_of_samples, int num_of_sets, float e) {
	GenerateSamples(num_of_samples, num_of_sets);

	MapSamplesToHemisphere(0);
}

__device__
void RegularSampler::GenerateSamples(int l_num_of_samples, int l_num_of_sets) {
	square_samples = (float2*)malloc(sizeof(float2) * l_num_of_samples * l_num_of_sets);
	int n = sqrt(float(l_num_of_samples));
	num_samples = n*n;
	num_sets = l_num_of_sets;
	for (int i = 0; i < l_num_of_sets; ++i)
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x)
				square_samples[i * num_samples + n * y + x] = make_float2(float(x)/n + 1.0/(2*n), float(y)/n + 1.0/(2*n));
}

__device__
void JitteredSampler::GenerateSamples(int l_num_of_samples, int l_num_of_sets) {

	square_samples = (float2*)malloc(sizeof(float2) * l_num_of_samples * l_num_of_sets);

	int n = sqrt(float(l_num_of_samples));
	num_samples = n*n;
	num_sets = l_num_of_sets;

	for (int i = 0; i < l_num_of_sets; ++i)
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x)
				square_samples[i * num_samples + n * y + x] = make_float2(float(x) / n + curand_uniform(&curand_state) / n, float(y) / n + curand_uniform(&curand_state) / n);
}


__device__
void MultiJitteredSampler::shuffle_coordinates(int offset) {
	for (int i = 0; i < num_samples; ++i) {
		int k = curand(&curand_state) % num_samples;
		int j = curand(&curand_state) % num_samples;
		float c;
		c = square_samples[i + offset].x;
		square_samples[i + offset].x = square_samples[k + offset].x;
		square_samples[k + offset].x = c;
		c = square_samples[i + offset].y;
		square_samples[i + offset].y = square_samples[j + offset].y;
		square_samples[j + offset].y = c;
	}
}
__device__
void MultiJitteredSampler::GenerateSamples(int l_num_of_samples, int l_num_of_sets) {
	square_samples = (float2*)malloc(sizeof(float2) * l_num_of_samples * l_num_of_sets);

	int n = sqrt(float(l_num_of_samples));
	num_samples = n*n;
	num_sets = l_num_of_sets;

	for (int i = 0; i < l_num_of_sets; ++i) {
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x)
				square_samples[i * num_samples + n * y + x] = make_float2(
					float(x) / n + (float(y) / n + curand_uniform(&curand_state) / n) / n,
					float(y) / n + (float(x) / n + curand_uniform(&curand_state) / n) / n);
		shuffle_coordinates(i * num_samples);
	}
}

__device__
void Sampler::MapSamplesToHemisphere(float e) {
	hemisphere_samples = (float3*)malloc(sizeof(float3) * num_samples * num_sets);

	for (int i = 0; i < num_samples * num_sets; ++i) {
		float3 tmp;
		float sin_phi = sin(2.0f * acos(-1.0f) * square_samples[i].x);
		float cos_phi = cos(2.0f * acos(-1.0f) * square_samples[i].x);
		float cos_theta = pow((1.0f - square_samples[i].y), 1.0f / (e + 1.0f));
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
		tmp = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
		hemisphere_samples[i] = tmp;
	}
}

__device__
float2 Sampler::SampleUnitSquare(int l_num_of_sample, int l_num_of_set) {
	return square_samples[num_samples * (l_num_of_set % num_sets) + (l_num_of_sample % num_samples)];
}

__device__
float3 Sampler::SampleHemisphere(int l_num_of_sample, int l_num_of_set) {
	return hemisphere_samples[num_samples * (l_num_of_set % num_sets) + (l_num_of_sample % num_samples)];
}

__device__
Sampler::~Sampler(void) {
	free(square_samples);
	free(hemisphere_samples);
}

__global__ void kernel(void) {
	curand_init(23041996, 0, 0, &curand_state);

	Sampler * test_sampler = new MultiJitteredSampler();
	test_sampler->init(4);

	for (int i = 0; i < test_sampler->num_samples; ++i) {
		float2 pt = test_sampler->SampleUnitSquare(i);
		float3 pts = test_sampler->SampleHemisphere(i);
		printf("(%f, %f), (%f, %f, %f)\n", pt.x, pt.y, pts.x, pts.y, pts.z);
	}

	delete test_sampler;
}