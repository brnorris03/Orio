void
mmul (float *A, int A_nrows, int A_ncols, float *B, int B_nrows, int B_ncols,
	  float *C, int C_nrows, int C_ncols)
{
	for (int i = 0; i < B_ncols; i += 1) {
		float *t4 = B + i * B_nrows;
		float *t6 = C + i * A_nrows;

		for (int ii = 0; ii < A_nrows; ++ii)
			t6[ii] = 0.0;
		for (int j = 0; j < A_ncols; j += 1) {
			float t8 = t4[j];
			float *t7 = A + j * A_nrows;

			for (int k = 0; k < A_nrows; k += 1) {
				float t10 = t7[k];
				float &t12 = t6[k];

				t12 += (t8 * t10);
			}
		}
	}
}
