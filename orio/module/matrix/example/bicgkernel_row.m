BICG
in
  A : matrix(row), p : vector(column), r : vector(column)
out
  q : vector(column), s : vector(column)
{
  q = A * p
  s = A' * r
}
