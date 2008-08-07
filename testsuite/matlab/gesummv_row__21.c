void GESUMMV(float* A, int A_nrows, int A_ncols, float* B, int B_nrows, int B_ncols, float a, float b, float* x, int x_nrows, float* y, int y_nrows) {
float* t7 = new float[B_nrows];
float* t8 = new float[B_nrows];
float* t3 = new float[A_nrows];
for (int i = 0; i < B_nrows; i+=1) {
float* t17 = B + i * B_ncols;
float& t19 = t7[i];
float t32 = 0;
for (int j = 0; j < B_ncols; j+=1) {
float t31 = x[j];
float t30 = t17[j];
t32 += (t30*t31);
}
t19 = t32;
}
for (int i = 0; i < B_nrows; i+=1) {
float t20 = t7[i];
float& t22 = t8[i];
t22 = (b*t20);
}
for (int i = 0; i < A_nrows; i+=1) {
float* t11 = A + i * A_ncols;
float& t13 = t3[i];
float t29 = 0;
for (int j = 0; j < A_ncols; j+=1) {
float t28 = x[j];
float t27 = t11[j];
t29 += (t27*t28);
}
t13 = t29;
}
for (int i = 0; i < A_nrows; i+=1) {
float t14 = t3[i];
float t24 = t8[i];
float& t26 = y[i];
t26 = (t24+(a*t14));
}
}
