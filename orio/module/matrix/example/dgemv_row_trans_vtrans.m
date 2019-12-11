DGEMV
in
  A : matrix(row), x : vector(column), alpha : scalar, beta : scalar
inout
  y : vector(row)
{
  y = alpha*(x'*A') + beta*y
}
