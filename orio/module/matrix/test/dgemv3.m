DGEMV
in
  A : matrix(column), x : vector(column), w : vector(column), e : vector(column)
inout
  y : vector(column), z : vector(column), p : vector(column)
{
  y = (A*x) + y
  z = (A*w) + z
  p = (A*e) + p
}
