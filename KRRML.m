function  accu=KRRML(tr_dat,tt_dat,trls,ttls,chit,cmiss,lambda)    
%Input:
%tr_dat  training set
%tt_dat  test set
%trls    training label
%ttls    test label
%chit    number of positive pair 
%cmiss   number of negative pair

%Output:
%accu    Classification accuracy
%get positive and negative sample pair
[zr,yr]=PPDpair(tr_dat,trls,chit,cmiss);
%metric learning;
% cmd=['-s 4 -t 5 -n ', num2str( nu),  ' -h 0 -f 1'];
[M]=KRR(tr_dat,yr,zr,lambda);
% [SVM_model,~]=svmtraintime(yr',zr',cmd);
%[SVM_model,~]=svmtraintime(yr',zr','-t 5 -h 0 -f 1');
%classification by the learned metric 
M=PosCone(M);
testSampleNum=size(tt_dat,2);
trainSampleNum=size(tr_dat,2);
predictLabels=zeros(size(ttls));
for i=1:testSampleNum
    dist=zeros(1,trainSampleNum);
    for j=1:trainSampleNum
        dist(j)=(tt_dat(:,i)-tr_dat(:,j))'*M*(tt_dat(:,i)-tr_dat(:,j));
    end
    [~,minindex]=mink(dist(:),chit);
    predictLabels(i)=mode(trls(minindex));
end
correctNum=length(find(predictLabels==ttls));
accu=correctNum/testSampleNum*100;
% accu=PPDclassify(SVM_model,tr_dat,tt_dat,trls,ttls,chit);
