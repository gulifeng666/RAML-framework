function [out] = acvation(x,type,flag)
if(flag==0)
if(strcmp(type,'sigmoid'))
out = sigmoid(x);
elseif(strcmp(type,'tanh'))
    out = tanh(x);
elseif(strcmp(type,'relu'))
    out = max(0,x);
elseif(strcmp(type,'sigmoidplus'))
    out = log(1+exp(x));
end
else
    if(strcmp(type,'sigmoid'))
out = sigmoid(x).*(1-sigmoid(x));
elseif(strcmp(type,'tanh'))
    out = 1-(tanh(x).^2);
elseif(strcmp(type,'relu'))
  
        x(x>0)=1;
        x(x<=0)=0;
    out=x;
elseif(strcmp(type,'sigmoidplus'))
    out = exp(x)./(1+exp(x));
    end
end

