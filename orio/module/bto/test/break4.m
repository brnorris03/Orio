GEMVER
in
  u1 : vector, u2 : vector, v1 : vector, v2 : vector, z1 : vector, z2 : vector
out
  B : column matrix
{
  C = u1 * v1' + u2 * v2'
  B = C + z1 * z2'
}
