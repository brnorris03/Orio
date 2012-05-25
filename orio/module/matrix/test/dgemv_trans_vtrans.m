DGEMV
in
  A : column matrix, x : vector, alpha : scalar, beta : scalar
inout
  y : row vector
{
  y = alpha*(x'*A') + beta*y
}
