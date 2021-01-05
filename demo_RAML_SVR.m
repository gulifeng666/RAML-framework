clear;
addpath('../common');
%%%%%%%%%%%%%%%     UCI    %%%%%%%%%%%%%%%%%%%%
load RAMLUCI;
dataset{1}=iris; 
dataset{2}=wpbc;
dataset{3}=wine; 
dataset{4}=sonar;
dataset{5}=glass;
dataset{6}=wdbc; 
dataset{7}=australian_credit;
%%%%%%%%%%%%%%%%    IMAGE   %%%%%%%%%%%%%%%%%%%%
load PCA100;
dataset{1}=binalpha1; 
dataset{2}=caltech101_silhouettes_16_1; 
dataset{3}=MnistData_05_1; 
dataset{4}=Mpeg7_1; 
dataset{5}=MSRA25_1; 
dataset{6}=news20_1; 
dataset{7}=TDT2_20_1; 
dataset{8}=uspst1; 
  
j=1;
% for lambda=[1e-3,1e-2,1e-1,1,10,100,1000];
method='pcsvr';%ncsvr
chit=1;
cmiss=3;
nu=0.5;
%lambdas=[10^-6,10^-5,10^-4,10^-3];
lambdas = [10^-2,10^-1,0,1,10];
% chit=1;
% cmiss=3;
cs = [0.5,0.1,0.2,0.5,1,2,5,10];
es=  [20,0.1,0.2,0.5,1,2,5,10,20,30];

for i=1:8
for p = 1:1
    for q = 1:1
  fprintf('%d-%d\n',p,q);  
  [acc_mean(j,p,q,:),acc_std(j,p,q,:)]=crossvalidate1(dataset{i},10,method,chit,cmiss,nu,0.001,3,0,'tanh',cs(p),es(q)); 
  %result=[acc_mean',acc_std'];
end
end
     save(['result',int2str(i),'.mat'],'acc_mean','acc_std'); 
j=j+1
end

