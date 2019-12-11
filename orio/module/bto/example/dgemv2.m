DGEMV
in
  A : column matrix, x : vector, w : vector
inout
  y : vector, z : vector
{
  y = (A*x) + y
  z = (A*w) + z
}
