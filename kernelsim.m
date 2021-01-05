function [KK]=kernelsim(Dataset1,Dataset2)
         deta=1;
         KK=zeros(size(Dataset1,1) ,size(Dataset2,1));
         for i = 1 : size(Dataset1,1)      
             for j=1:size(Dataset2,1)
               KK(i,j)=exp(-(norm(Dataset1(i,:)-Dataset2(j,:)))^2/(2*deta*deta)); %£¨¸ßË¹ºË£©
             end
         end
end