DOT2
in
	x : vector(column), y : vector(column), z : vector(column)
out
	alpha : scalar, beta : scalar
{
	alpha = x' * y
	beta = x' * z
}
