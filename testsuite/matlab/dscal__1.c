void
DSCAL (float *A, int A_nrows, int A_ncols, float a, float *B, int B_nrows,
	   int B_ncols)
{
	for (int i = 0; i < A_ncols; i += 1) {
		float *t4 = A + i * A_nrows;
		float *t6 = B + i * A_nrows;

		for (int j = 0; j < A_nrows; j += 1) {
			float t7 = t4[j];
			float &t9 = t6[j];

			t9 = (a * t7);
		}
	}
}
