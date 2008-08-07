void
AXPYDOT (float a, float *u, int u_nrows, float *v, int v_nrows, float *w,
		 int w_nrows, float &r, float *z, int z_nrows)
{
	float t19 = 0;

	for (int i = 0; i < w_nrows; i += 1) {
		float t10 = v[i];
		float t13 = w[i];
		float &t16 = z[i];
		float t18 = u[i];

		t16 = (t13 - (a * t10));
		t19 += (t18 * t16);
	}
	r = t19;
}
