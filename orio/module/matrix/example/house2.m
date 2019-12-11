AATX
in
 A : matrix(column), v : vector(column), alpha : scalar
out 
 B : matrix(column)
{
  x = alpha*v*(v'*A)
 B = A - x
}
