DGEMVT
in
  A : matrix(column), y : vector(column), z : vector(column), a : scalar, b : scalar
out
  x : vector(column), w : vector(column)
{
  x = b * (A' * y) + z
  w = a * (A * x)
}
