GEMVER
in
  u1 : vector(column), u2 : vector(column), v1 : vector(column), v2 : vector(column)
out
  B : matrix(column)
{
  B = u1 * v1' + u2 * v2'
}
