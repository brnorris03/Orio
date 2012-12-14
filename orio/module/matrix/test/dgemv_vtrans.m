DGEMV
in
  A : matrix(column), x : vector(column), alpha : scalar, beta : scalar
inout
  y : vector(row)
{
  y = alpha*(x'*A) + beta*y
}
