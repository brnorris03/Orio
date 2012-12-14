DGEMV
in
  A : matrix(orientation=column), x : vector(orientation=column), alpha : scalar, beta : scalar
inout
  y : vector(orientation=column)
{
  y = alpha*(A*x) + beta*y
}
