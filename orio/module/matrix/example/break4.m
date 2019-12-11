GEMVER
in
  u1 : vector(column), u2 : vector(column), v1 : vector(column), 
	v2 : vector(column), z1 : vector(column), z2 : vector(column)
out
  B : matrix(column)
{
  C = u1 * v1' + u2 * v2'
  B = C + z1 * z2'
}
