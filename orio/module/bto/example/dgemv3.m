DGEMV
in
  A : column matrix, x : vector, w : vector, e : vector
inout
  y : vector, z : vector, p : vector
{
  y = (A*x) + y
  z = (A*w) + z
  p = (A*e) + p
}
