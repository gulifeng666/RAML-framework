function distance_index = muti_label_sampling(pairsize,L,samplinglambda,distance_prob)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%������ֲ���ʳ���
%input��
%samplinglambda ϵ�������Ƴ�������ƽ���̶�
%distance_prob   ��������
%pairsize        ������Ŀ
%output��
%distance_index  �õ�����������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


      n = length(L);
      distance_index = zeros(n,n);
      %�ж���û���ṩ��������
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
      %û��ָ���������ʵĻ�ʹ������֮�����ĵ�����Ϊ��������
      for i =1:n          
           [A,I] = sort(L(i,:));
           sampling_prob= 1./(A(2:n)+0.000001);
           %���Ƴ������ʵ�ƽ���̶�
           sampling_prob  = sampling_prob.^samplinglambda;
           if(i~=n)
               sampling_prob(i) = 0;
           end
           sampling_prob = sampling_prob/sum(sum(sampling_prob));
          
           distance_index(i,1:1+pairsize) = randsrc(1,pairsize+1,[2:n;sampling_prob]);
      end
      end
end

