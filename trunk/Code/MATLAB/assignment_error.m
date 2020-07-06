K=randi([5,25]);
N=randi([500000,1000000]);
G=rand(K,N);
p=rand(K,1);
p=p./sum(p);
q=stablepoint(G,p,10,'diff');%time hog!
CMG=costMatrix(G,q);%time hog!
[~,idxBig]=max(squeeze(CMG(1,2:N+1,1:K)),[],2);
target=round(q*N,0);
uvidx=unique(idxBig);
received = histc(idxBig,uvidx);
labelerror=(received-target)./target
targeterror=(target-ones(K,1)*round(N/K))*K/N
round(mean(idxBig)-K/2)
max(labelerror)
min(labelerror)
% .82 rel error seed: 132588448