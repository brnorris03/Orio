AxATy
in
A : matrix(orientation=row), x : vector(orientation=column), y : vector(orientation=column)
out
  w : vector(orientation=column), z : vector(orientation=column)
{
  w = A * x
  z = A' * y
}
