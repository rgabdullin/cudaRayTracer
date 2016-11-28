#ifndef __OrthographicCamera__
#define __OrthographicCamera__

#include "Camera.h"

class OrthographicCamera : public Camera {
public:
	__device__
	OrthographicCamera(World *wr);
	__device__
	void MakeRay(Ray* ray, int2 pixel, float2 sample);
};

#endif