
function distance_index= getbatch(X,y,L,pairsize,flag,type)
%flag=pairsize 多标记学习不平衡，单标记学习不平衡
%flag =pairsize*2 多标记学习多类平衡
%other多标记学习平衡程度。单标记控制平衡程度
sim = 0;
dis = 1;
n = ceil(pairsize/2);
distance_index = zeros(size(X,1),size(X,1));
sortvalue = sort(L(:));
meanvalue = sortvalue(int16(length(L(:)/2)));
class_size =2;
if(strcmp(type,'classify'))
    
    for i=1:length(y)
        [A,I] = sort( sum((X(i,:)-X).^2,2));
        samindex = find(L(i,I)==sim); 
        disindex = find(L(i,I)==dis);
        samlength = length(samindex);
        disindex = disindex(1:samlength);
        samindex  =samindex( randi(length(samindex),[n+1+flag,1]));
        disindex   =disindex( randi(length(samindex),[pairsize-flag-n,1]));
        distance_index(i,1:n+1+flag) = samindex;
        distance_index(i,n+flag+2:pairsize+1) = disindex;
    return
end
if(strcmp(type,'label_dis'))
    L(L<=0.5)=sim;
    L(L>0.5)=dis;
end
if(strcmp(type,'muti_label'))
    L(L<=0.5)=sim;
    L(L==0.5)=dis;
end
if(strcmp(type,'muti_label')&&flag==pairsize*2)
maxvalue = max(L(:));
for i =1:size(X,1)
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
for i =1:size(X,1)
  if(flag~=pairsize)
    [A,I] = sort( sum((X(i,:)-X).^2,2)); 
    simindex = find(L(i,I)==sim);
    [A,I] = sort( sum((X(i,:)-X).^2,2)); 
    disindex = find(L(i,I)==dis);
    if(length(simindex)>=n+1+flag)
    distance_index(i,1:n+1+flag) = I(simindex(1:n+1+flag));
    
    else
        if(length(simindex)>0)
         distance_index(i,1:length(simindex)) = I(simindex(1:length(simindex)));
         distance_index(i,length(simindex)+1:n+1+flag) = I(2:2+n+flag-length(simindex));
        else
    distance_index(i,1:n+1+flag) = I(simindex(1:n+1+flag));
        end
    end
    if(length(disindex)<=pairsize-flag-n)
        if(length(disindex)>0)
         distance_index(i,n+flag+2:n+2+flag+length(disindex)-1) = I(disindex(1:length(disindex)));
         distance_index(i,n+flag+2+length(disindex):pairsize+1) = I(2:pairsize-length(disindex)-flag-n+1);
        end
        distance_index(i,n+flag+2:pairsize+1) = I(2:pairsize-flag-n+1);
    else
         distance_index(i,n+flag+2:pairsize+1) = I(disindex(1:pairsize-flag-n));
    end
    
  else
    [A,I] = sort( sum((X(i,:)-X).^2,2)); 
    distance_index(i,:) = I;
  end          
end
end
end

