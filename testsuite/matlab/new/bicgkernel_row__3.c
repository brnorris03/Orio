void BICG(float* A, int A_nrows, int A_ncols, float* p, int p_nrows, float* r, int r_nrows, float* q, int q_nrows, float* s, int s_nrows) {
for (int ii = 0; ii < s_nrows; ++ii)
s[ii] = 0.0;
for (int i = 0; i < A_nrows; i+=1) {
float t12 = r[i];
float* t8 = A + i * A_ncols;
float& t10 = q[i];
float t16 = 0;
for (int j = 0; j < A_ncols; j+=1) {
float t15 = p[j];
float t14 = t8[j];
float& t19 = s[j];
t19 += (t12*t14);
t16 += (t14*t15);
}
t10 = t16;
}
}
