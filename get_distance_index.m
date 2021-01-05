function distance_index= get_distance_index(X,y,L,pairsize,flag,type)
%���룺X: ��������
%      y: ��ǩ
%      L: ��ǩ��ϵ����
%      pairsize: ������Ŀ
%      flag: ���Ƴ����̶ȱ�ǩ 
%            flag == pairsize ���ڳ�����ȡ�����
%            flag == pairsize*2  ����ѧϰ����ƽ�����
%            flag == other       ����ƽ��̶ȳ���,eg:flag=0,����������Ŀһ��
%      type: ѧϰ����ʽ,classify/muti_label
%�����distance_index  ����������������

sim = 0;
dis = 1;
n = ceil(pairsize/2);
distance_index = zeros(size(X,1),size(X,1));
class_size =2;
m = size(X,1);    
%���������ֵ�Ļ���ֵ������������
if((strcmp(type,'muti_label')&&(max(L(:))<=1)))
    L(L<=0.5)=sim;
    L(L>=0.5)=dis;
end
%�������ɢֵ���Ҳ���ƽ������Ļ���ֵ��
if(strcmp(type,'muti_label')&&(max(L(:))>1)&&(flag~=pairsize*2))
    L(L<=mean(L,2))=sim;
    L(L>mean(L,2))=dis;
end
   
%����ѧϰ����ƽ�����
if(strcmp(type,'muti_label')&&flag==pairsize*2)
maxvalue = max(L(:));
for i =1:m
    [A,I] = sort( sum((X(i,:)-X).^2,2));
    k=2;
   index=find(L(i,I)==0);
   if(length(index)>class_size)
    distance_index(i,1:1+class_size) = index(1:1+class_size);
   else
    distance_index(i,1:1+class_size) = I(k:k+class_size);
    k=k+class_size;
   end
for j =1:maxvalue
    index = find(L(i,I)==j);
    
    if(length(index)>=class_size)
     distance_index(i,(j)*class_size+2:class_size*(j+1)+1)=index(1:class_size);
    else
     distance_index(i,(j)*class_size+2:(j+1)*class_size+1)=I( k:k+class_size-1);
     k=k+class_size;
    
    end
end
end
   
 
else  
for i =1:m
  %����ƽ��̶ȳ���,eg:flag=0,�����һ��������˵ȡ��������������һ��  
  if(flag~=pairsize)
    [A,I] = sort( sum((X(i,:)-X).^2,2)); 
    simindex = find(L(i,I)==sim);
    disindex = find(L(i,I)==dis);
    if(length(simindex)>=n+1+flag)
    distance_index(i,1:n+1+flag) = I(simindex(1:n+1+flag));
    else
        if(length(simindex)>0)
         distance_index(i,1:length(simindex)) = I(simindex(1:length(simindex)));
         distance_index(i,length(simindex)+1:n+1+flag) = I(2:2+n+flag-length(simindex));
        else
          distance_index(i,1:n+1+flag) = I(simindex(1:n+1+flag));%���Ƴ̶�
        end
    end
    if(length(disindex)<=pairsize-flag-n)
        if(length(disindex)>0)
         distance_index(i,n+flag+2:n+2+flag+length(disindex)-1) = I(disindex(1:length(disindex)));
         distance_index(i,n+flag+2+length(disindex):pairsize+1) = I(2:pairsize-length(disindex)-flag-n+1);
        end
        distance_index(i,n+flag+2:pairsize+1) = I(2:pairsize-flag-n+1);
    else
        if(strcmp(type,'classify'))
         classset = unique(y);
         disclassset = setdiff(classset,y(i));
         disclassindex = randi(length( disclassset),[pairsize-flag-n,1]);
        % disindex = [];
         for p =1:pairsize-flag-n
             disflag = y(I)==disclassset(disclassindex(p));
             tmp =  I(disflag);
             tmpindex = randi(length(tmp),1);
             tmp = tmp(tmpindex);
       %      disindex=[disindex  tmp(1)];
         end
       %  distance_index(i,n+flag+2:pairsize+1) = disindex;
          distance_index(i,n+flag+2:pairsize+1)=  I(disindex(1:pairsize-flag-n));    
        else
         distance_index(i,n+flag+2:pairsize+1) = I(disindex(1:pairsize-flag-n));
        end
    end
    
  else
     %���ڳ���,flag==pairsize,ȡ�����pairsize��
    [A,I] = sort( sum((X(i,:)-X).^2,2)); 
   distance_index(i,1:pairsize+1) = I(1:pairsize+1);
  end          
end
end
end
