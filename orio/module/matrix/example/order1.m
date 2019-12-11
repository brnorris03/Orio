order
in
A : matrix(column), x : vector(column), B : matrix(column)
inout
  y : vector(column)
{
  y = A * (B * x)
}
