%% Approximate the convergence rate of grad log iteration
%% Very good candidate for parallelization
rng('shuffle');
max=2^32-1;
seeds=randi(max,1,100);

s=0;
dat=zeros(1000,1);
mu=zeros(100,1);
vars=mu;
h=waitbar(s/100,sprintf('On sample %d of 100',s));
for j=.1:.1:10
    s=s+1;
    rng(seeds(s))
    waitbar(s/100,h,sprintf('On sample %d of 100',s));
    tic
    for i=1:1000
        [F,p]=GMMData(5,5*round(10^j),2,randi([2^16, max]));
        pHat=stablepoint(F,ones(5,1)./5,12,'diff');
        dat(i)=norm(p'-pHat);
    end
    toc
    mu(s)=mean(dat);
    vars(s)=var(dat);
end