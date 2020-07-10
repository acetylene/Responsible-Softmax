%% Inital values setup
K=20;
N=1000;
sampleSize = 10000;
%Filename has rng seed in it.
dir=strcat(pwd,'\errors data\');

rng('shuffle');
scurr = rng;

filename=strcat(dir,'gmm_errors_',num2str(K),'_',num2str(N),'_',num2str(scurr.Seed),'.csv');
%% Generate data
%currently does GMM data.
seeds=randi([1000000,90000000],1,sampleSize);
errors=zeros(K,sampleSize);
D=2;
init=ones(1,K)./K;
W=waitbar(1/sampleSize,sprintf('Calculating Sample %d of %d',1,sampleSize));
for i=1:sampleSize
    [F,P]=GMMData(K,N,D,seeds(i));
    Phat=stablepoint(F,P',12);
    errors(:,i)=P-Phat';
    waitbar((i+1)/sampleSize,W,sprintf('Calculating Sample %d of %d',i+1,sampleSize));
end


%% File I/O
%should check to be sure filename isn't already there.
csvwrite(filename,errors');