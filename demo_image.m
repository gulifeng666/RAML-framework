datalist{1}='emotion';
datalist{2}='flags';
% load Flower-102_VGG %Standford-40_VGG,Flower-102_VGG,CUB-200-2011_VGG,Caltech-256_VGG
Samples=tr_descr';
Labels=tr_label';
testS=tt_descr';
testL=tt_label';
chit=1;
cmiss=3;
nu=0.5;
method='RML';%'ITML''GMML''LMNN''RML''DOUBLESVM'
 if(strcmp(method,'ITML'))
    % datai=data(:,1:column-1);
   
%     save datai.mat datai;
    addpath('.\ITML')
   [ClassRate,fangcha]=Test(Samples,Labels,testS,testL);
    acc_u=ClassRate;
    acc_std=fangcha;
end
 switch method
%         case 'PPDML'
%             ClassRate=PPDML(Samples',testS',Labels,testL,chit,cmiss);  
%         case 'knn'
%             ClassRate=ldmlknn(Samples,Labels,testS,testL,chit,dtype); 
%         case 'RMLv'
%             nupool=[0.05 0.1 0.15 0.2 0.3 0.4 0.5];
%             for i=1:7
%               rate(i)=crossvalidate([Samples,Labels],fold,'RMLv',chit,cmiss,nupool(i));  
%             end
%             [value,index]=max(rate);
%        
%             ClassRate=RML(Samples',testS',Labels,testL,chit,cmiss,nupool(index)); 
        case 'RML'
             ClassRate=RML(Samples',testS',Labels,testL,chit,cmiss,nu); 
        case 'GMML'
            addpath('.\GMML')
             ClassRate=gmml(Samples,Labels,testS,testL,chit,cmiss);
        case 'LMNN'
            addpath('.\LMNN')
            ClassRate=runlmnn(Samples,Labels,testS,testL);
        case 'DOUBLESVM'
            addpath('.\DOUBLESVM')
            ClassRate=doublesvm(Samples,Labels,testS,testL);
    end
currentFolder = pwd;
addpath(genpath(currentFolder));
    
% method='emotion';
method='MLKNN';
data_select=[1 4 6 7];  
chit=3;

 for i=1:length(data_select)
    kk=data_select(i);
    eval(['load ' [datalist{kk} '_train']])
    eval(['load ' [datalist{kk} '_test']])
    fprintf(datalist{kk});

    % iindex{k,i}=fea_ind;

     switch method
       case 'MLKNN'
           M=MLML(train_data',train_target',chit); 
           %M=eye(size(train_data,2));
          [HammingLoss(i),RankingLoss(i),OneError(i),Coverage(i),Average_Precision(i)]=mlknn(train_data,train_target,test_data,test_target,M);
     end
end
      
result=[HammingLoss',RankingLoss',OneError',Coverage',Average_Precision'] 
