function err=kernel_ridge_regression(tr_dat,tt_dat,tr_lab,tt_lab)


    lambda=1e-3;
    for i=1:length(tr_lab)
       for j=1:length(tr_lab)
          ktr(i,j)=kernelsim(tr_dat(i),tr_dat(j)); 
       end
    end
    alpha=tr_lab'*pinv(ktr+lambda*eye(size(ktr,1)));
    for i=1:length(tr_lab)
        for j=1:length(tt_lab)
           ktt(i,j)=kernelsim(tr_dat(i),tt_dat(j)); 
        end
    end
    yy=alpha*ktt;
    err=mean(power((yy'-tt_lab),2));
end