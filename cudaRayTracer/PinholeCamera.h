#ifndef __PinholeCamera__
#define __PinholeCamera__

#include "Camera.h"

class PinholeCamera : public Camera {
	float dist_to_vp;
	float zoom;
public:
	__device__
	PinholeCamera(World *wr);
	
	__device__
	void init_pinhole(float l_dist_to_vp, float l_zoom = 1.0f);
	__device__
	void MakeRay(Ray* ray, int2 pixel, float2 sample);
};

#endif