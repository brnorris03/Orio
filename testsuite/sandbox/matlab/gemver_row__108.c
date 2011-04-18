void GEMVER(float* A, int A_nrows, int A_ncols, float a, float b, float* u1, int u1_nrows, float* u2, int u2_nrows, float* v1, int v1_nrows, float* v2, int v2_nrows, float* y, int y_nrows, float* z, int z_nrows, float* B, int B_nrows, int B_ncols, float* w, int w_nrows, float* x, int x_nrows) {
float* t14 = new float[A_ncols];
for (int i = 0; i < A_nrows; i+=1) {
float t31 = u2[i];
float t24 = u1[i];
float* t27 = A + i * A_ncols;
float* t37 = B + i * A_nrows;
for (int j = 0; j < A_ncols; j+=1) {
float t61 = v2[j];
float t54 = v1[j];
float t57 = t27[j];
float& t67 = t37[j];
t67 = ((t57+(t24*t54))+(t31*t61));
}
}
for (int ii = 0; ii < A_ncols; ++ii)
t14[ii] = 0.0;
for (int i = 0; i < A_nrows; i+=1) {
float t39 = y[i];
float* t38 = B + i * A_ncols;
for (int j = 0; j < A_ncols; j+=1) {
float t68 = t38[j];
float& t70 = t14[j];
t70 += (t39*t68);
}
}
for (int i = 0; i < A_ncols; i+=1) {
float t41 = t14[i];
float t45 = z[i];
float& t47 = x[i];
t47 = (t45+(b*t41));
}
for (int i = 0; i < A_nrows; i+=1) {
float* t48 = B + i * A_ncols;
float& t53 = w[i];
float t73 = 0;
for (int j = 0; j < A_ncols; j+=1) {
float t72 = x[j];
float t71 = t48[j];
t73 += (t71*t72);
}
t53 = (a*t73);
}
}
