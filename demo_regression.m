addpath('./data');
addpath('./LDL');
addpath('./common');
% datalist{1}='emotion';
% datalist{2}='flags';
  datalist{3}='wine';
  datalist{2}='parkinsons';
  datalist{1}='housing';
  datalist{4}='crime';
  datalist{5}= 'forestfires';
result = [];
type = 'HRAML';
pairsize =4;
bachsize =4;
methodnum = 5;

result=[];
lambdas = [0.0001,0.0001,0.00001];

knnresult=[];
for q=1:3
for p = 1:1
for i=1:1
    % iindex{k,i}=fea_ind;
    eval(['load ' [datalist{i}]])
     switch type
          case 'MLKRR-PSD1'
              for c =1:10
                  for e =0.1:0.1:1
                      MLMLresult = [];
                       for j =1:5
                             n= length(X);
             left = (j-1)*int16(n*0.2)+1;
             right = min(j*int16(n*0.2),n);
             testindex =left:right;
             Xtest = X(testindex,:);
             ytest = Y(testindex,:);
             index = 1:n;
             trainindex = index>right|index<left;
             Xtrain = X(trainindex,:);
             ytrain = Y(trainindex,:);
              M = MLML_PSD(Xtrain',ytrain,chit,c,e,'1'); 
              MLMLresult = [MLMLresult ;knn(Xtrain,ytrain,Xtest,ytest,4,'regression',M,0)];
             
                       end
                        result = [result;mean(MLMLresult)];
                  end
              end
                case 'MLKRR-PSD2'
              for c =1:10
                  for e =0.1:1:0.1
              M = MLML_PSD(train_data',train_target',chit,c,e,'2'); 
              MLMLresult = [MLMLresult ;knn(Xtrain,ytrain,Xtest,ytest,4,'regression',M,0)];
              result = [result;mean(MLMLresult)];
                  end
              end
         case 'RML-KRR'
              M = MLMLKRR(Xtrain',ytrain,4,0.5);
              MLMLresult = [MLMLresult ;knn(Xtrain,ytrain,Xtest,ytest,4,'regression',M,0)];
              result = [result;mean(MLMLresult)];
         case 'HRAML'  
             iter=100;
            balenced = 0;
             debug= 1;% 0：不输出
            alafa = 0;% 0：不进行取平均
            useval = 0;
            layers =3;
            hidden = 10;
            acthidden = 'tanh';
            actout ='tanh';
            normalize = 1;
             n= length(X);
             rng(3);
             index =randperm(n,n);
             X =X(index,:);
             Y = Y(index,:);
             HRMLresult = [];
             MLMLresult = [];
             for j =1:5
             left = (j-1)*int16(n*0.2)+1;
             right = min(j*int16(n*0.2),n);
             testindex =left:right;
             Xtest = X(testindex,:);
             ytest = Y(testindex,:);
             index = 1:n;
             trainindex = index>right|index<left;
             Xtrain = X(trainindex,:);
             ytrain = Y(trainindex,:);
             knnresult = [knnresult ;knn(Xtrain,ytrain,Xtest,ytest,4,'regression',0,0)];
            HRMLresult = [HRMLresult;HRAML(Xtrain,Xtest,ytrain,ytest,lambdas(q),pairsize,'regression',iter,balenced,bachsize,debug,alafa,useval,pairsize,layers,hidden,acthidden,actout,normalize,p*0.3)];
            end
            %result =[result;mean(knnresult),std(knnresult);mean(MLMLresult),std(MLMLresult)];  
            result = [result;mean(HRMLresult)];
             end
     end
end
end
save('matlabresult','result');
     

