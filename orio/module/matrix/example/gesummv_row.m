GESUMMV
in
  A : matrix(row),
  B : matrix(row),
  x : vector(column),
  a : scalar,
  b : scalar
out
  y : vector(column)
{
  y = a * (A * x) + b * (B * x)
}
