function [acc_mean,acc_std]=crossvalidate1(data,fold,method,chit,cmiss,nu,lambda,l,p,func,c,e)
[row column]=size(data);
pairsize =4;
bachsize  =40;
methodnum = 5;
type = 'classify';
iter=20;
balenced = 0;
debug= 0;% 0：不输出
alafa = 0;% 0：不进行取平均
useval = p;%是否用验证
layers =l;%层数
hidden = 100;%隐层数目
hiddenact =func;%激活函数
outact =func;%输出激活函数
label=data(:,column);
classnum=max(label);
start1=1;
for i=1:classnum
    [a,b]=find(label==i);
    datai=data(a,:);      %select the i class data 
    [rr1,cc1]=size(datai);
    start1=1;
    %%%%%%%%%part the i class in (fold)%%%%%%%%%%%%%%%%%%%%%
    for j=1:fold-1
        a1=round(length(a)/fold);
        a2=a1-1;
        %fun1=strcat('x*',num2str(a1),'+y*',num2str(a2),'=',num2str(rr1)); 
        %fun2=strcat('x+y=',num2str(fold)); 
        %[x,y]=solve(fun1,fun2) 
        %[x,y] = solve('x*a1+a2*y=rr1','x+y=fold')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        A=[a1 a2;1 1];
        b=[rr1 fold]';
        x=A\b;
        if (j<x(1)+1)
            everynum=a1;
        else
            everynum=a2;
        end
        start2=start1+everynum-1;       
        eval(['data' num2str(i) num2str(j) '=datai([start1:start2],:);']);
        start1=start2+1;
    end
    eval(['data' num2str(i) num2str(fold) '=datai([start1:length(a)],:);']);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:fold
    eval(['part' num2str(j) '=[];']);
    for i=1:classnum
      eval(['part' num2str(j) '=[part' num2str(j) ';data' num2str(i) num2str(j) '];']);
    end   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:fold
    Samples=[];
     Labels=[];
     testS=[];
     testL=[];
    for i=1:fold
        
        if (i~=j)
            eval(['Samples=[Samples;part' num2str(i) '(:,1:column-1)];'])
            eval(['Labels=[Labels;part' num2str(i) '(:,column)];'])
        end
    end
    eval(['testS=part' num2str(j) '(:,1:column-1);'])
    eval(['testL=part' num2str(j) '(:,column);'])
    switch method
        case 'PPDML'
            ClassRate=PPDML(Samples',testS',Labels,testL,chit,cmiss);  
        case 'knn'
            ClassRate=ldmlknn(Samples,Labels,testS,testL,chit,dtype); 
        case 'RMLv'
            nupool=[0.05 0.1 0.15 0.2 0.3 0.4 0.5];
            for i=1:7
              rate(i)=crossvalidate([Samples,Labels],fold,'RMLv',chit,cmiss,nupool(i));  
            end
            [value,index]=max(rate);
       
            ClassRate=RML(Samples',testS',Labels,testL,chit,cmiss,nupool(index)); 
        case 'RML'
             ClassRate=RML(Samples',testS',Labels,testL,chit,cmiss,nu); 
        case 'pcsvr'
             ClassRate=RML_SVR(Samples',testS',Labels,testL,chit,cmiss,c,e,'pcsvr');    
        case 'ncsvr'
             ClassRate=RML_SVR(Samples',testS',Labels,testL,chit,cmiss,c,e,'ncsvr');      
         case 'HRAML'
             ClassRate= HRAML(Samples,testS,Labels,testL,lambda,pairsize,type,iter,balenced,bachsize,debug,alafa,useval,0,layers,hidden,hiddenact,outact,1);
          
        case 'KRRML'
            ClassRate=KRRML(Samples',testS',Labels,testL,chit,cmiss,lambda)

    end
    accu_m(j,:)=ClassRate 
    if(j==9)
        1
    end
end
acc_mean=mean(accu_m)
acc_std=std(accu_m)
