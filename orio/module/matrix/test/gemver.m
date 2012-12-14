GEMVER
in
  A : matrix(column), u1 : vector(column), u2 : vector(column), 
  v1 : vector(column), v2 : vector(column),
  a : scalar, b : scalar,
  y : vector(column), z : vector(column)
out
  B : matrix(column), x : vector(column), w : vector(column)
{
  B = A + u1 * v1' + u2 * v2'
  x = b * (B' * y) + z
  w = a * (B * x)
}
