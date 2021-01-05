function acc =hraml_predict(xtrain,ytrain,xtest,ytest,pairsize,type,method,layers,flag,theta,hiddenact,outact,debug,normalize)
%addpath('G:/matlab/PSDML/LDL');
if(flag)
theta = load('hraml_theta.mat');
elseif(~flag&&strcmp(type,'classify'))
    classset = unique(ytest);
    classnum = length(classset);
    pclassnum = zeros(length(ytest),1);
    maxnum=0;
    for k =1:classnum
        num = sum( ytest == classset(k));
        pclassnum( ytest == classset(k)) = num;
       if(maxnum<num)
           maxnum = num;
       end
    end
    pclassnum=double(maxnum./pclassnum);
end
 for la=1:layers-2
     xtrain = acvation(xtrain*theta.W{la,1}'+theta.b{la,1}',hiddenact,0);
     xtest = acvation(xtest*theta.W{la,1}'+theta.b{la,1}',hiddenact,0);
 end
   xtrain = acvation(xtrain*theta.W{layers-1,1}'+theta.b{layers-1,1}',outact,0);
   for i =1:length(xtrain)
       if(normalize)
       xtrain(i,:) = xtrain(i,:)/norm(xtrain(i,:),2);
       end
   end
     xtest = acvation(xtest*theta.W{layers-1,1}'+theta.b{layers-1,1}',outact,0);
     for i =1:size(xtest,1)  
         if(normalize)
       xtest(i,:) = xtest(i,:)/ norm(xtest(i,:),2);
         end
     end

X = [xtrain;xtest];
if(strcmp(type,'classify')||strcmp(type,'regression'))
    predictLabels = zeros(size(ytest,1),1);
    if(strcmp(method,  'knn'))

for i=1:size(ytest,1)
    dist=zeros(1,size(ytrain,1));
    for j=1:size(xtrain,1)
        dist(j)=(xtest(i,:)-xtrain(j,:))*(xtest(i,:)-xtrain(j,:))';
    end
    %[~,minindex]=mink(dist(:),chit);
    [A,I]= sort(dist);
    if(strcmp(type,'classify'))
    predictLabels(i)=mode(ytrain(I(1:pairsize)));
    else
    predictLabels(i)=mean(ytrain(I(1:pairsize)));
    end     
end 
else
    n = size(xtrain,1);
    m = size(xtest,1);
    distance = zeros(m,n);
for j =1:m
        distance(j,:) = sum((xtest(j,:)-xtrain).^2,2)';
end
if(min(ytrain==0))
    ytrain=ytrain+1;
    ytest = ytest+1;    
end
K = max(ytrain);
res = zeros(m,K);
    for k =1:K
        for i = 1:m
        res(i,k) = mean(distance(i,ytrain==k))/mean(distance(i,ytrain~=k));
        end
    end
   [A ,predictLabels] = min(res,[],2); 
    end
end
if(strcmp(type,'classify'))
    if(~flag)
        acc = sum((predictLabels==ytest).*pclassnum)/(classnum*maxnum);
        return;
    end
acc=length(find(predictLabels==ytest))/length(ytest);
elseif(strcmp(type,'regression'))
   acc=[sum((predictLabels-ytest).^2)/length(ytest),sum(abs(predictLabels-ytest))/length(ytest)];
elseif(strcmp(type,'muti_label'))
acc= mlknn(xtrain,ytrain',xtest,ytest',eye(size(xtrain,2),size(xtrain,2)));
else 
acc  = aaknn(xtrain,ytrain,xtest,ytest,'L2',eye(size(xtrain,2),size(xtrain,2)));     
end
end