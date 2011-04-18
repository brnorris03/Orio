void
vadd (float *w, int w_nrows, float *y, int y_nrows, float *z, int z_nrows,
	  float *x, int x_nrows)
{
	for (int i = 0; i < w_nrows; i += 1) {
		float t7 = y[i];
		float t6 = w[i];
		float t11 = z[i];
		float &t13 = x[i];

		t13 = (t11 + (t6 + t7));
	}
}
