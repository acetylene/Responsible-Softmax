function [GMMDistValues,P, data] = GMMData(K,N,D,seed)
%% Create GMM distribution parameters from the given seed
%% K-Number of Clusters, N-sample size, D-dimension of data
rng(seed,'twister');
Mu=rand(K,D);
Sigma=zeros(D,D,K);
for i=1:K
    Sigma(:,:,i)=full(sprandsym(D,(rand(1)+2)/4,rand(D,1)));
end
s=rand(1,K);
P=s./sum(s);

%% Create gmdistribution object, and a sample
gm = gmdistribution(Mu,Sigma,P);
X=random(gm,N);
data = X;

%% Create parameter matrix F as evaluation of PDFs at sample.
GMMDistValues =zeros(K,N);
for i=1:K
    GMMDistValues(i,:)=mvnpdf(X,Mu(i,:),Sigma(:,:,i));
end
end