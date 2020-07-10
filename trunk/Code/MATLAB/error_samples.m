%% Inital values setup
K=20;
N=1000;
sampleSize = 10000;
%The use of pwd requires that this script be run from the containing
%folder.
dir=strcat(pwd,'\errors data\');
%Shuffle chages the seeds every run. Set to 'default' or some other seed
%if consistent results are required
rng('shuffle');
scurr = rng;
%Filename has rng seed in it.
filename=strcat(dir,'gmm_errors_',num2str(K),'_',num2str(N),'_',num2str(scurr.Seed),'.csv');
%% Generate data
%currently creates GMM data.
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
%CSVWRITE checks to be sure filename isn't already there.
% default behavior if the file exists is to overwrite data.
csvwrite(filename,errors');