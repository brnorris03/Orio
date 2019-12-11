mgs
in
  q : vector(column)
inout 
 v : vector(column)
{
  r = q' * v
  v = v - r*q
}
