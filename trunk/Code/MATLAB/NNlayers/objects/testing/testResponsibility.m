clear; clc;
K = 5;%randi(16)+4;%number of classes
N = 10000;%Total sample size
TestSz = .3; %size of verification sample

err = 1e-6;
tolH = tolCheckerHilb(err);
tolE = tolCheckerEuc(err);
respH = responsibilityOperator(K,tolH);
respE = responsibilityOperator(K,tolE);
assert(tolH.satisfiedBy(respH.pi_0,respE.pi_0))
assert(tolE.satisfiedBy(respH.pi_0,respE.pi_0))

rng('default')
rng(102404)

%% GMM for sampling
%try two equdistributed samples with different means. Try different mixing.

% mu = randi(20,K,1); %means
mu = (linspace(-10*K,10*K,K))';
sigma = rand(1,1,K)/2+1; %variances
pStar = rand(1,K); 
pStar = pStar./sum(pStar);%mixture coefficients
pStar = mean([ones(1,K)./K;pStar]);%pull the weights closer to even
gm = gmdistribution(mu, sigma, pStar);%gmm

[X,T] = random(gm,N);%total sample

F = zeros(K,N);
for i= 1:K
    F(i,:) = normpdf(X,mu(i),squeeze(sigma(1,1,i)));
end

orbitH5 = respH.iteratedResp(F,respH.pi_0,5);
orbitE5 = respE.iteratedResp(F,respE.pi_0,5);

orbFullH = respH.fixedResp(F,respH.pi_0);
orbFullE = respE.fixedResp(F,respE.pi_0);