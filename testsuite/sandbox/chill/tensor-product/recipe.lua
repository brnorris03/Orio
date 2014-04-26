init("_orio_chill_.c","*u,",0)
dofile("cudaize.lua")

N = 10
nelt = 100
distribute({0,1,2},1)

tile_by_index(0,{"i"},{2},{l1_control="ii"},{"l","j","ii","i","k"})CU=1
tile_by_index(1,{"k"},{2},{l1_control="kk"},{"l","j","i","kk","k","m"})
tile_by_index(2,{"i"},{2},{l1_control="ii"},{"l","j","ii","i","k"})CU=1
cudaize(0,"tensor1_GPU",{u=nelt*N*N*N,ur=nelt*N*N*N,us=nelt*N*N*N,ut=nelt*N*N*N,D=N*N,Dt=N*N},{block={"l","j"}, thread={"ii","i"}},{})CU=3


cudaize(1,"tensor2_GPU",{u=nelt*N*N*N,ur=nelt*N*N*N,us=nelt*N*N*N,ut=nelt*N*N*N,D=N*N,Dt=N*N},{block={"l","j"}, thread={"kk","k","i"}},{})CU=4
cudaize(2,"tensor3_GPU",{u=nelt*N*N*N,ur=nelt*N*N*N,us=nelt*N*N*N,ut=nelt*N*N*N,D=N*N,Dt=N*N},{block={"l","j"}, thread={"ii","i"}},{})CU=5

