POWMETHOD
in
	A : matrix(column), b : vector(column)
out
	d : vector(column)
{
	c = A*b
	d = (c* c') * c
}

