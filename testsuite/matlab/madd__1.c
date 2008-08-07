void
madd (float *A, int A_nrows, int A_ncols, float *B, int B_nrows, int B_ncols,
	  float *C, int C_nrows, int C_ncols)
{
	for (int i = 0; i < A_ncols; i += 1) {
		float *t5 = B + i * B_nrows;
		float *t4 = A + i * A_nrows;
		float *t7 = C + i * A_nrows;

		for (int j = 0; j < A_nrows; j += 1) {
			float t9 = t5[j];
			float t8 = t4[j];
			float &t11 = t7[j];

			t11 = (t8 + t9);
		}
	}
}
