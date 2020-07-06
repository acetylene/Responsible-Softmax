clear;
clc;

%% Setup of basic parameters
rng(10072019);
K = randi(16)+4;%number of classes
N = 8192;%Total sample size
TestSz = .3; %size of verification sample

%% GMM for sampling
mu = randi(20,K,1); %means
sigma = rand(1,1,K)/2+1; %variances
pStar = rand(1,K); 
pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm

%% Setup the matrix F with 'correct' parameters
[Y,C] = random(gm,N);
F = zeros(K,N);
for j=1:N
for i=1:K
F(i,j) = normpdf(Y(j),gm.mu(i),squeeze(gm.Sigma(:,:,i)));
end
end

%% Guessing mixture parameters
p_0 = 1/K.*ones(K,1);

% Averaging mixtures
numTests = 64;
c = cvpartition( N, 'KFold', numTests );

[pHatsSm{numTests + 1},orbitsSm{numTests + 1}] = stablepoint(F,p_0,12,'diff',true);
[pHatsBig{numTests + 1},orbitsBig{numTests + 1}] = deal(pHatsSm{numTests + 1},orbitsSm{numTests + 1});
 
for ii = 1:numTests
    [pHatsSm{ii},orbitsSm{ii}] = stablepoint(F(:,c.test(ii)),p_0,12,'diff',true);
    [pHatsBig{ii},orbitsBig{ii}] = stablepoint(F(:,c.training(ii)),p_0,12,'diff',true);
end

pHatSmAvg = mean(cat(2,pHatsSm{1:numTests}),2);
pHatBigAvg = mean(cat(2,pHatsBig{1:numTests}),2);

pHat1 = pHatsSm{numTests+1};

sprintf('Norm of pHat-pStar is %d.', norm(pHat1-pStar'))
sprintf('Norm of pHat-smAvg is %d.', norm(pHat1-pHatSmAvg))
sprintf('Norm of pHat-bigAvg is %d.', norm(pHat1-pHatBigAvg))

