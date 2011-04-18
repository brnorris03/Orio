format long;
nreps=4;
sizes=[500, 1000,10000,20000];
tm=zeros(size(sizes));
i=1;
for reps = 1:nreps
  tm(i) = 0;
  for n = sizes(1:4)
    A=rand(n,n);
    B=rand(n,n);
    b=rand(1,1);
    a=rand(1,1);
    y=rand(n,1);
    u1=rand(n,1);
    u2=rand(n,1);
    v1=rand(n,1);
    v2=rand(n,1);
    x=zeros(n,1);
    z=zeros(n,1);
    w=zeros(n,1);
  
    t=cputime;
    B = A + u1 * v1' + u2 * v2';
    x = b * (B' * y) + z;
    w = a * (B * x);
    tm(i) = tm(i)+ cputime - t;
    i = i + 1;
  
    clear A;
    clear B;
    clear w;
    clear x;
    clear y;
    clear z;
    clear u1;
    clear u2;
    clear v1;
    clear v2;
  end
  tm(i) = tm(i)/nreps
end
