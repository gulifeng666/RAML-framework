function demo_PPDML(dataname,chit,cmiss)

%Demo for point to point distance metric learning

%Input:
%dataname  name of the input data
%chit      number of positive pairs per sample 
%cmiss     number of negative pairs per sample 

%usage     demo_PPDML('Sonar',1,3)

load(dataname); %load the data set
sampleNum=size(X,2);
accu=zeros(1,10);
% 10-fold cross validation
for i=1:10
    disp(strcat('Training the fold ', num2str(i)));
    trainLabel=setdiff(1:sampleNum,kfold{i});
    testLabel=kfold{i};
    tr_dat=X(:,trainLabel);
    tt_dat=X(:,testLabel);
    trls=Y(trainLabel);
    ttls=Y(testLabel);
    accu(i)=PPDML(tr_dat,tt_dat,trls,ttls,chit,cmiss);    
end
rate=mean(accu);
fprintf('\n')
fprintf(['recogniton rate of PPDML is ' num2str(rate)]);
fprintf('\n')