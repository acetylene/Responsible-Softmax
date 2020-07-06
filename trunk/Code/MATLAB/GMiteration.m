N = 10000;
K = 5;
mu = (1:K)';
sigma = .5;
%rng('default');
truep = rand(1,5);
truep =truep./sum(truep);
gm = gmdistribution(mu, sigma, truep);
for i=K:-1:1
    F{i} = @(x) normpdf(x, mu(i), sigma);
end

X = random(gm,N);
G = zeros(K,N);
for i=1:K
    G(i,:)=[F{i}(X)];
end

p=stablepoint(G,ones(K,1)./K,12,'diff')';
