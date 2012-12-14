AATX
in
  A : matrix(orientation=column), x : vector(orientation=column)
out
  y : vector(orientation=column)
{
  y = A * (A' * x)
}
