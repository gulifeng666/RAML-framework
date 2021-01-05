
%%%%%%%%%%%%%%%%    IMAGE   %%%%%%%%%%%%%%%%%%%%
 load PCA100;
% MnistData_05_1(:,1:100)=(MnistData_05_1(:,1:100)-mean(MnistData_05_1(:,1:100)))./std(MnistData_05_1(:,1:100));
%  MSRA25_1(:,1:100)=(MSRA25_1(:,1:100)-mean(MSRA25_1(:,1:100)))./std(MSRA25_1(:,1:100)); 
%  Mpeg7_1(:,1:100)=(Mpeg7_1(:,1:100)-mean(Mpeg7_1(:,1:100)))./std(Mpeg7_1(:,1:100));
 dataset{1}=binalpha1; 
 dataset{2}=caltech101_silhouettes_16_1; 
 dataset{3}=MnistData_05_1; 
 dataset{4}=Mpeg7_1; 
 dataset{5}=MSRA25_1; 
 dataset{6}=news20_1; 
 dataset{7}=TDT2_20_1; 
 dataset{8}=uspst1; 


 
method='HRAML';%KRRML%DRML
chit=1;
cmiss=3;
nu=0.5;
lambda=[0.01,0.001,0.0001];
for p =1:1
for l =1:2  %layers
    for j =1:2%Ñ§Ï°ÂÊ
for i=1:8
    [acc_mean(p,l,j,i,:),acc_std(p,l,j,i,:)]=crossvalidate1(dataset{i},10,method,chit,cmiss,nu,lambda(j),l+1,p,'tanh'); 
end
    end
end
end
result=[acc_mean',acc_std']; 
result=result/100
