function [GMMDistValues,P, data] = GMMData(K,N,D,seed)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
rng(seed,'twister');
Mu=rand(K,D);

Sigma=zeros(D,D,K);
for i=1:K
    Sigma(:,:,i)=full(sprandsym(D,(rand(1)+2)/4,rand(D,1)));
end

s=rand(1,K);
P=s./sum(s);
gm = gmdistribution(Mu,Sigma,P);

X=random(gm,N);
data = X;
GMMDistValues =zeros(K,N);

for i=1:K
    GMMDistValues(i,:)=mvnpdf(X,Mu(i,:),Sigma(:,:,i));
end

end

