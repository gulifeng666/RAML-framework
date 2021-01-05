function M =myquadrog(yr,zr,c,e,type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%type  优化类型   svr/pcsvr/ncsvr
%Output:
%M     度量矩阵
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(zr,1);
%yr = yr+1;
m = size(zr,2)/2;
%裁剪过大的数据
if(n>6000)
    rand_index =  randi(n,[6000,1]);
    yr = yr(rand_index,:);
    zr = zr(rand_index,:);
    n = 6000;
end
dim=m;
K = zeros(n,n);
Yold = rand(m,m);
Y =zeros(m,m);
x = rand(2*n,1);
for i =1:n 
        K(i,:) = ((sum((repmat(zr(i,1:dim)-zr(i,dim+1:end),n,1)).*(zr(:,1:dim)-zr(:,dim+1:end)),2)).^2)'; 
end

H = zeros(size(zr,1)*2,size(zr,1)*2);
H(1:n,1:n) = K;
H(n+1:2*n,n+1:2*n)=K;
H(1:n,1+n:2*n)= -K;
H(1+n:2*n,1:n)=-K;
switch(type)
    case'svr'
ky = zeros(n,1);
for i =1:n
    ky(i)=(zr(i,1:dim)-zr(i,dim+1:end))*Y*((zr(i,1:dim)-zr(i,dim+1:end))');
end
D = eig(Y);
%sum(sum(D<0))
f = e*ones(2*n,1)+[-yr;yr]+[ky;-ky];
Aeq=zeros(n*2,n*2);
Aeq(1:n,1:n)=eye(n,n);
Aeq(1:n,n+1:2*n)=-eye(n,n);
obj1 = 0.5*(x'*H*x)+(f'*x);
A = [ones(n,1);-ones(n,1)];

Aeq = zeros(2*n,2*n);
Aeq(1:n,1:n)=eye(n);
Aeq(1:n,n+1:end)=-eye(n);
Aeq = Aeq.*([yr;yr]');
yr1 = [yr;yr];
Aeq = zeros(2*n,2*n);
for i =1:10:n
    Aeq(i,[i,i+1,i+2,i+3,i+n,i+n+1,i+n+2,i+n+3])=[yr1([i,i+1,i+2,i+3]);-yr1([i+n,i+n+1,i+n+2,i+n+3])];
end    
%x = quadprog(H,f,[],[],[],zeros(2*n,1),zeros(2*n,1),ones(2*n,1)*c);
x = quadprog(H,f,[],[],[],[],zeros(2*n,1),ones(2*n,1)*c);
x_index = find((x(1:n,1)-x(n+1:2*n,1))~=0);
Y = zeros(m,m);
for i =1:size(x_index,1)
    index = x_index(i);
    Y = Y-(x(index)-x(index+n))*((zr(index,1:dim)-zr(index,dim+1:end))'*(zr(index,1:dim)-zr(index,dim+1:end)));
end
Y = PosCone(Y);
obj2  = 0.5*(x'*H*x)+(f'*x);
    case 'pcsvr'
f = e*ones(2*n,1)+[-yr;yr];
Aeq=zeros(n*2,n*2);
Aeq(1:n,1:n)=-eye(n,n);
Aeq(1:n,n+1:2*n)=eye(n,n);
%obj1 = 0.5*(x'*H*x)+(f'*x);
x = quadprog(H,f,Aeq,zeros(2*n,1),[],[],zeros(2*n,1),ones(2*n,1)*c);
%obj2  = 0.5*(x'*H*x)+(f'*x);
Y = zeros(m,m);
    case'ncsvr'
        iter = 0;
        maxiter = 5;
        s = zeros(n,1); 
        beta = zeros(n,1);
        beta_old = zeros(n,1);
        x = zeros(2*n,1);
        s_old = zeros(n,1);
        x_old = zeros(2*n,1);
while(1)
beta = (s-(x(1:n)-x(n+1:2*n)));
kr = K*beta;
f = e*ones(2*n,1)+[-yr;yr]+[kr;-kr];
x_old = x;
x = quadprog(H,f,[],[],[],[],zeros(2*n,1),c*ones(2*n,1),[]);
s_old = s;
beta_old = beta;
%beta = quadprog(K,(x(1:n)-x(n+1:2*n))'*K,[],[],[],[],zeros(n,1),[],[]);
s = quadprog(K,-K*(x(1:n)-x(n+1:2*n)),[],[],[],[],zeros(n,1),[],[]);
sum1 = sum(abs(x_old-x)); 
%sum2 = sum(abs(s_old-s));
sum2 = sum(abs(beta_old-beta));
disp([sum1,sum2]);
if(iter==maxiter|(sum1<10^-10&sum2<10^-10))
    break;
end
iter=iter+1;
%if(sum(abs(obj2-obj1))<10^-4)
%    break;
%end
Yold = Y; 
end
end
if(type=='ncsvr')
    M  = zeros(m,m);
    beta = s-(x(1:n)-x(n+1:2*n));
  mu = beta+(x(1:n)-x(n+1:2*n));
   % s = K*beta;
    %mu = beta+(x(1:n)-x(n+1:2*n));
   
    x_index = find(mu~=0);
    for i =1:size(x_index,1)
    index = x_index(i);
    M = M+mu(i)*((zr(index,1:dim)-zr(index,dim+1:end))'*(zr(index,1:dim)-zr(index,dim+1:end)));
    end
else
x_index = find((x(1:n)-x(n+1:2*n))~=0);
M = zeros(m,m);
for i =1:size(x_index,1)
    index = x_index(i);
    M = M+(x(index)-x(index+n))*((zr(index,1:dim)-zr(index,dim+1:end))'*(zr(index,1:dim)-zr(index,dim+1:end)));
end
M = M+Y;
end
%dist = zeros(n,1);
%for i =1:n
%   dist(i) = ((zr(i,1:dim)-zr(i,dim+1:end))*M*(zr(i,1:dim)-zr(i,dim+1:end))');
%end
%disp([min(dist-yr),max(dist-yr),mean(dist-yr)]);
end
