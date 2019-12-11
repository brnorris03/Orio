CGPRECOND
in
A : matrix(column), B : matrix(column)
inout
p : vector(column), z : vector(column), beta : scalar, r : vector(column), 
x : vector(column)
{
p = z + beta * p
w = A * p
dpi = p' * w
alpha = beta * dpi
x = x + alpha * p
r = r - alpha * w
z = B * r
beta = z' * r 
}
