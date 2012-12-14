GESUMMV
in
  A : matrix(column),
  B : matrix(column),
  x : vector(column),
  a : scalar,
  b : scalar
out
  y : vector(column)
{
  y = a * (A * x) + b * (B * x)
}
