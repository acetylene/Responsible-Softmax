function [p] = gmOneD(N,K,truep)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
mu = (1:K)';
sigma = .5;

gm = gmdistribution(mu, sigma, truep);
for i=K:-1:1
    F{i} = @(x) normpdf(x, mu(i), sigma);
end

X = random(gm,N);
G = zeros(K,N);
for i=1:K
    G(i,:)=F{i}(X);
end

p=stablepoint(G,ones(K,1)./K,12,'diff')';
end

