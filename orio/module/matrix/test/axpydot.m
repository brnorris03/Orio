AXPYDOT
in
  w : vector(column), a : scalar, v : vector(column), u : vector(column)
out
  z : vector(column), r : scalar
{
  z = w - a * v
  r = z' * u
}
