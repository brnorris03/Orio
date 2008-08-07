void
BICG (float *A, int A_nrows, int A_ncols, float *p, int p_nrows, float *r,
	  int r_nrows, float *q, int q_nrows, float *s, int s_nrows)
{
	for (int ii = 0; ii < q_nrows; ++ii)
		q[ii] = 0.0;
	for (int i = 0; i < A_ncols; i += 1) {
		float t9 = p[i];
		float *t8 = A + i * A_nrows;
		float &t13 = s[i];
		float t19 = 0;

		for (int j = 0; j < A_nrows; j += 1) {
			float t17 = r[j];
			float t14 = t8[j];
			float &t16 = q[j];

			t19 += (t17 * t14);
			t16 += (t9 * t14);
		}
		t13 = t19;
	}
}
