DGEMV
in
  A : matrix(column), x : vector(column), w : vector(column)
inout
  y : vector(column), z : vector(column)
{
  y = (A*x) + y
  z = (A*w) + z
}
