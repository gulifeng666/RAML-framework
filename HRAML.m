function acc= HRAML(X,Xtest,y,ytest,lambda,pairsize,type,T,balenced,bachsize,debug,alafa,useval,flag,layers,hidden,hiddenact,outact,normalize,samplinglambda)
%输入：  X:   训练样本
%       Xtest: 测试样本
%       y:    训练标签     
%       ytest: 测试标签
%       lambda:学习率 
%       pairsize:抽样数目
%       type:    学习方式
%       T:       总迭代次数
%       balenced: 正负平衡
%       bachsize
%       debug:   调试
%       alafa:    正负加权
%       useval:   使用验证
%       flag:      抽样标志,控制抽样方法和程度
%       layers   
%       hidden    
%       hiddenact
%       outact
%       normalize
%       samplinglambda: 概率抽样系数
%输出： acc 准确率
n = size(X,1);
rng(3);
%随机划分数据
index =randperm(n,n);
X =X(index,:);
y = y(index,:);
h = [size(X,2)];
%隐藏层数目
for l =1:layers-2
    h = [h hidden];
end
dim = size(X,2);
h = [h,dim];
[Xtrain,L,Ltrain,Xval,Lval,ytrain,yval] = process_train_data(X,y,type,useval);
% if(strcmp(type,'muti_label'))
% distance_prob = distanceprob(Ltrain);
% end
%得到抽样索引
distance_index = get_distance_index(Xtrain,ytrain,Ltrain,pairsize,flag,type);
if(useval)
%flag=pairsize
val_distance_index = get_distance_index(Xval,yval,Lval,pairsize,pairsize,type);
end
dis =1;
sim=0;
neborL = get_neborL(distance_index,pairsize,Ltrain);
if(useval)
val_neborL = get_neborL(val_distance_index,pairsize,Lval);
end
[index_x1,index_y1] = find(neborL==2);
[index_x2,index_y2] = find(neborL==0);
patience = 300;
n1 = size(index_x1,1);
n2 = size(index_x2,1);
%L(randi(100,20,1),randi(100,20,1))=-1;
n  = size(Ltrain,1);
val_n = size(Lval,1);
class_weight = ones(n,pairsize);
val_class_weight = ones(val_n,pairsize);

if(alafa~=0) 
class_weight(neborL==dis) =alafa /sum(sum(neborL==dis));
class_weight(neborL==sim) = 1/sum(sum(neborL==sim));
class_weight(neborL==dis) =alafa /sum(sum(neborL==dis));
class_weight(neborL==sim) = 1/sum(sum(neborL==sim));
val_class_weight(val_neborL==dis) =alafa /sum(sum(val_neborL==dis));
val_class_weight(val_neborL==sim) = 1/sum(sum(val_neborL==sim));
val_class_weight(val_neborL==dis) =alafa /sum(sum(val_neborL==dis));
val_class_weight(val_neborL==sim) = 1/sum(sum(val_neborL==sim));
end
%class_weight(L==1)=sum(sum(L==-1))/sum(sum(L==1));

%X = eye(n);
%X = (X-mean(X(:)))/std(X(:));
distance = zeros(n,pairsize);  
val_distance = zeros(val_n,pairsize);  

b = cell(layers-1,1);
W = inittheta(h,1);
for la=1:layers-1
     b{la,1}= zeros(h(la+1),1);
end
Wgrad = cell(size(W));
bgrad = cell(size(b));
mw = cell(size(W));
mb = cell(size(b));
for la = 1:layers-1
    Wgrad{la,1} = zeros(size(W{la,1}));
    bgrad{la,1} = zeros(size(b{la,1}));
    mw{la,1} = zeros(size(W{la,1}));
    mb{la,1} = zeros(size(b{la,1}));
end

bestresult =[1, 10^9];
error = zeros(T,1);
val_error = zeros(T,1);

beta =0.0001;
if(debug)
acc = zeros(4,T);
end
t =0;
for t =1:T
switch type
    case 'regression'
    %distance_index = get_distance_index(Xtrain,ytrain,Ltrain,pairsize,pairsize,type);
    distance_index = muti_label_sampling(pairsize,Ltrain,samplinglambda,0);
    case 'label_dis'
  %  distance_index = get_distance_index(Xtrain,ytrain,Ltrain,pairsize,pairsize,type);
    distance_index = muti_label_sampling(pairsize,Ltrain,samplinglambda,0);
    case 'muti_label'
    %distance_index = get_distance_index(Xtrain,ytrain,Ltrain,pairsize,pairsize,type);
    distance_index = muti_label_sampling(pairsize,Ltrain,samplinglambda,0);
end
  for i =1:n*pairsize
   
    if(rem(i,2)==0)
        if(balenced)
    j = randi(n1,1);
    t1 =index_x1(j);
    t2 = distance_index(t1,index_y1(j)+1);
        else
    [t1,t2] = get_pair_index(i,distance_index,pairsize);
        end
%     t1 = randi(n,1);
%     t2 = randi(n,1);
    else
        if(balenced)
    j = randi(n2,1);
    t1 =index_x2(j);
   t2 = distance_index(t1,index_y2(j)+1);
        else
    [t1,t2] = get_pair_index(i,distance_index,pairsize);
        end
   % t1 = randi(n,1);
  %  t2 = randi(n,1);
    end
    %正传播
     a = cell(layers,2);
    z = cell(layers,2); 
    s = cell(layers,2);  
   a{1,1} =  Xtrain(t1,:)';
   a{1,2} =  Xtrain(t2,:)';
    
    for la = 2:layers-1
    z{la,1} = W{la-1,1}*a{la-1,1}+b{la-1,1};
    a{la,1} = acvation(z{la,1},hiddenact,0);
    end 
    for la = 2:layers-1
    z{la,2} = W{la-1,1}*a{la-1,2}+b{la-1,1};
    a{la,2} = acvation(z{la,2},hiddenact,0);
    end 
    z{layers,1} = W{layers-1,1}*a{layers-1,1}+b{layers-1,1};
    a{layers,1} = acvation(z{layers,1},outact,0);
     z{layers,2} = W{layers-1,1}*a{layers-1,2}+b{layers-1,1};
    a{layers,2} = acvation(z{layers,2},outact,0);
    if(normalize)
   a{layers,1}=a{layers,1}/norm(a{layers,1},2);
   a{layers,2}=a{layers,2}/norm(a{layers,2},2);
    end
   % if((t>1||i>pairsize*5)&&(sum((a{layers,1}-a{layers,2}).^2)>1 &&Ltrain(t1,t2)==1))
  %     continue
 %   end
    %求梯度，反向传播
    grad =(sum((a{layers,1}-a{layers,2}).^2)-Ltrain(t1,t2))*(a{layers,1}-a{layers,2});
    if(normalize)
    normgrad1 = eye(dim)/norm(a{layers,1},2)-(a{layers,1}*a{layers,1}'/(norm(a{layers,1},2)^3));
    normgrad2 = eye(dim)/norm(a{layers,2},2)-(a{layers,2}*a{layers,2}'/(norm(a{layers,2},2)^3));
    else
    normgrad1=1;
    normgrad2=1;
    end
    s{layers,1} = normgrad1*grad.*acvation(z{layers,1},outact,1);
    s{layers,2} = normgrad2*grad.*acvation(z{layers,2},outact,1);
    for la = layers-1:-1:2
    s{la,1} = W{la,1}'*s{la+1,1}.*(acvation(z{la,1},hiddenact,1));
    s{la,2} = W{la,1}'*s{la+1,2}.*(acvation(z{la,2},hiddenact,1));
    end 
    for la = 1:layers-1
    Wgrad{la,1} = Wgrad{la,1}+s{la+1,1}*(a{la,1}')-s{la+1,2}*(a{la,2}');
    bgrad{la,1} = bgrad{la,1}+s{la+1,1}-s{la+1,2};
    end

  
    if(rem(i,bachsize) ==0)
      e = (T-t)/T;
      e =1;
      for la = 1:layers-1
    Wgrad{la,1} = Wgrad{la,1}+bachsize/n/pairsize*beta*W{la,1};
    bgrad{la,1} = bgrad{la,1}+bachsize/n/pairsize*beta*b{la,1};
     end
      
   
      for la = 1:layers-1
    mw{la,1} = mw{la,1}*0.9+Wgrad{la,1};
    mb{la,1} = mb{la,1}*0.9+bgrad{la,1};
      end
    
       for la = 1:layers-1
    W{la,1} = W{la,1}-e*lambda*mw{la,1};
    b{la,1} = b{la,1}-e*lambda*mb{la,1};
      end
    
       for la = 1:layers-1
    Wgrad{la,1} = zeros(size(Wgrad{la,1}));
    bgrad{la,1} = zeros(size(bgrad{la,1}));
     end
    end
    
  end
 %预测     
 predict = Xtrain;
 if(useval)
 val_predict = Xval;
 end
 for la=1:layers-2
     predict = acvation(predict*W{la,1}'+b{la,1}',hiddenact,0);
     if(useval)
     val_predict = acvation(val_predict*W{la,1}'+b{la,1}',hiddenact,0);
     end
 end
  predict = acvation(predict*W{layers-1,1}'+b{layers-1,1}',outact,0);
  if(useval)
     val_predict = acvation(val_predict*W{layers-1,1}'+b{layers-1,1}',outact,0);
  end
if(t==1)
 save('hraml_theta.mat','W','b');
end
    for j =1:n
        distance(j,:) = sum((predict(j,:)-predict(distance_index(j,2:pairsize+1),:)).^2,2)';
    end
    if(useval)
     for j =1:size(Xval,1)
        val_distance(j,:) = sum((val_predict(j,:)-val_predict(val_distance_index(j,2:pairsize+1),:)).^2,2)';
     end
    end
    error(t) = 0.25*sum(sum(class_weight.*(distance-neborL).^2));
    if(useval)
    val_error(t) = 0.25*sum(sum(val_class_weight.*((val_distance-val_neborL).^2)));
    end
    for la=1:layers-1
      error(t)=error(t)+0.5*beta*(sum(sum(W{la,1}.^2))+sum(sum(b{la,1}.^2)));
      if(useval)
      val_error(t)=val_error(t)+0.5*beta*(sum(sum(W{la,1}.^2))+sum(sum(b{la,1}.^2)));
      end
    end
    if(alafa==0)
    error(t)=error(t)/n/pairsize;
    if(useval)
    val_error(t) = val_error(t)/val_n/pairsize;
    end
    end
    theta = struct();
    theta.W = W;
    theta.b = b;
    if(useval==0)  
    a = hraml_predict(Xtrain,ytrain,Xtest,ytest,pairsize,type,'knn',layers,0,theta,hiddenact,outact,debug,normalize) ;
    if(strcmp(type,'classify'))
      error(t)=1-a;
    elseif(strcmp(type,'muti_label'))
            error(t)=1-a(5);
    elseif(strcmp(type,'label_dis'))
            error(t) = a(3);
    else
         error(t)=a(1);
       
    end
   if(error(t)<bestresult(2) )
     % if(t==1||(val_error(t)<val_error(max(1,t-1))-(10^-5)))
       
       bestresult = [t,error(t)];
       save('hraml_theta.mat','W','b');
   end
    
    else
     a = hraml_predict(Xtrain,ytrain,Xval,yval,pairsize,type,'knn',layers,0,theta,hiddenact,outact,debug,normalize) ; 
    if(strcmp(type,'classify'))
       val_error(t)=1-a;
    elseif(strcmp(type,'muti_label'))
            val_error(t)=1-a(5);
    elseif(strcmp(type,'label_dis'))
            val_error(t) = a(3);
    else
         val_error(t)=a(1);
    end
    if(val_error(t)<bestresult(2)-(10^-5))
     % if(t==1||(val_error(t)<val_error(max(1,t-1))-(10^-5)))
       save('hraml_theta.mat','W','b');
       bestresult = [t,val_error(t)];
    end
    end
    
    fprintf('iteration:%d       train_loss:%f\n',t,error(t));
    fprintf('best:%d,%f val_loss:%f\n',bestresult(1),bestresult(2),val_error(t));
  
    if(t-bestresult(1)>patience)
       break;
    end
     if(t==3||t==2 &&error(t)>error(t-1))
       % lambda = lambda*0.1;
     elseif(t>3&&(error(t-1)-error(t-2))>0&&error(t-1)-error(t-3)>0)    
  %    lambda = lambda*0.9;
      % lambda = lambda*0.1;
     end
    
end
 acc = hraml_predict(X,y,Xtest,ytest,pairsize,type,'knn',layers,1,0,hiddenact,outact,0,normalize) ;
% fprintf(' train_loss_dis:%f\n',bestresult(1),bestresult(2)-error(max(1,bestresult(1)-1)));
% save(['G:/data/nnet_log/',[[type,mat2str(notuseval),mat2str(rat),mat2str(layers),mat2str(hidden),hiddenact,mat2str(lambda),mat2str(samplinglambda)],'.mat']],'error','val_error'); 
end
%predict =sign(w3'* [1;tanh(w2'*[1;tanh(w1'*X')])]);
function [t1,t2] = get_pair_index(i,distance_index,pairsize)

if(rem(i,pairsize)==0)
    t1 = floor(i/pairsize);
    t2 = distance_index(t1,pairsize+1);
else
    t1 = floor(i/pairsize)+1;
    t2 = distance_index(t1,rem(i,pairsize)+1);
end

end

function result = logis_loss(x)
       beta1 = 1;
       result =1/beta1*log(1+exp(beta1*x));
end
function grad = logis_grad(x)
          beta1=1;
          grad = exp(beta1*x)/(1+exp(beta1*x));
end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function grad = sigmoid_grad(x)
        grad = sigmoid(x).*(1-sigmoid(x));
end
function grad =tanh_grad(x)
        grad = 1-tanh(x).^2;
end
function neborL = get_neborL(distance_index,pairsize,L)
n = size(L,1);
neborL = zeros(n,pairsize);
for i =1:n
    
    neborL(i,:) = L(i,distance_index(i,2:pairsize+1));
end
end                                                                                                   



   

