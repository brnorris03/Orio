DGEMV
in
  A : row matrix, x : vector, alpha : scalar, beta : scalar
inout
  y : vector
{
  y = alpha*(A*x) + beta*y
}
