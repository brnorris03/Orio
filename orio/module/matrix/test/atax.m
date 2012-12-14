ATAX
in
  A : matrix(orientation=row), x : vector(orientation=column)
out
  y : vector(orientation=column)
{
  y = A' * (A * x)
}
