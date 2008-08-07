void
WAXPBY (float a, float b, float *x, int x_nrows, float *y, int y_nrows,
		float *w, int w_nrows)
{
	for (int i = 0; i < x_nrows; i += 1) {
		float t11 = y[i];
		float t8 = x[i];
		float &t17 = w[i];

		t17 = ((a * t8) + (b * t11));
	}
}
