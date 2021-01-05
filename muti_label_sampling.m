function distance_index = muti_label_sampling(pairsize,L,samplinglambda,distance_prob)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%多任务分层概率抽样
%input：
%samplinglambda 系数，控制抽样概率平滑程度
%distance_prob   抽样概率
%pairsize        抽样数目
%output：
%distance_index  得到的样本索引
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


      n = length(L);
      distance_index = zeros(n,n);
      %判断有没有提供抽样概率
      if(distance_prob~=0)
      n = size(distance_prob,1);
      sampling_prob = zeros(n,n);
      for i =1:n          
           sampling_prob(i,:) = (distance_prob(i,:)).^samplinglambda;
           sampling_prob(i,i) = 0;
           sampling_prob(i,:) = sampling_prob(i,:)/sum(sampling_prob(i,:));
           distance_index(i,1:pairsize+1) =  randsrc(1,pairsize+1,[1:n;sampling_prob(i,:)]);
      end
      else
      %没有指定抽样概率的话使用样本之间距离的倒数作为抽样概率
      for i =1:n          
           [A,I] = sort(L(i,:));
           sampling_prob= 1./(A(2:n)+0.000001);
           %控制抽样概率的平滑程度
           sampling_prob  = sampling_prob.^samplinglambda;
           if(i~=n)
               sampling_prob(i) = 0;
           end
           sampling_prob = sampling_prob/sum(sum(sampling_prob));
          
           distance_index(i,1:1+pairsize) = randsrc(1,pairsize+1,[2:n;sampling_prob]);
      end
      end
end

