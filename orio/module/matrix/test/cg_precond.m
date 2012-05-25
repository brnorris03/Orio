CGPRECOND
in
A : matrix, B : matrix
inout
p : vector, z : vector, beta : scalar, r : vector, x : vector
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
