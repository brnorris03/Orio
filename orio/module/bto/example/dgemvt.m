DGEMVT
in
  A : column matrix, y : vector, z : vector, a : scalar, b : scalar
out
  x : vector, w : vector
{
  x = b * (A' * y) + z
  w = a * (A * x)
}
