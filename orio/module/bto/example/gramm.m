mgs
in
  q : vector
inout 
 v : vector
{
  r = q' * v
  v = v - r*q
}
