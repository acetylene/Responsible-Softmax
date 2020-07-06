clear;
clc;

rng(1085456391); %for testing

K = 5;%randi(16)+4;%number of classes
N = 10000;%Total sample size
D = 2;
TestSz = .3; %size of verification sample

%% GMM for sampling
mu = [linspace(-K,K,K)*3.5;randi(4,1,K)-2]'; %means
density = 1;
rc = .9;
for k=K:-1:1
    x = rand();
    if mod(k,2) == 0 
        A = [0,-1;1,0];
        evals = [5*x,x];
    else
%         tmp = sprand(D,D,density,rc);
        A = -eye(D);%[1,-1;1,1]./sqrt(2);%full(tmp);
        evals = [19*x,x];
    end
    
    Sig = diag(evals);
    
    Sigma(:,:,k) = A*Sig*A' ; %variances
end

%adjust for specific example
mu(2,1) = -9;
mu(4,1) = 9;
Sigma(:,:,4) = Sigma(:,:,2);
Sigma(:,:,5) = Sigma(:,:,1);

idx = [1,3,5];%unique(randi(K,1,K-1));
pStar = rand(1,K);
pStar(idx) = pStar(idx)+10;

pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, Sigma, pStar);%gmm

% rng('shuffle');
% scurr = rng;
rng(1271026191);
[X,T] = random(gm,N);%total sample

figure
scatter(X(:,1),X(:,2),10,T)

%% Sensitivity analysis wrt Parameters
for i=K:-1:1
F(i,:) = mvnpdf(X,mu(i,:),Sigma(:,:,i));
end

p_0 = ones(K,1)./K;
pHat_F = stablepoint(F,p_0,12,'diff',false);

%peturbation of F
precision = 1;
err = 10^-precision;
numRuns = 2048;
W = waitbar(0,sprintf('Calculating Run %d of %d',0,numRuns));
for i = numRuns:-1:1
    waitbar((numRuns+1-i)/numRuns,W,...
            sprintf('Calculating Sample %d of %d',(numRuns+1-i),numRuns))
    G = F.*exp(randn(K,N)*err);%how to add positive error?
    tic
    pHats_G(:,:,i) = stablepoint(G,p_0,12,'diff',false);
    t = toc;
    if t>1
        extremeG{i} = G;
    end
end
close(W);

diffs_sg = pStar'-squeeze(pHats_G);
diffs_fg = pHat_F-squeeze(pHats_G);

figure; 
subplot(2,2,1); histogram(sum(abs(diffs_fg)),'Normalization','pdf')
Y = sqrt(diag(diffs_fg'*diffs_fg));
subplot(2,2,2); histogram(Y,'Normalization','pdf')
subplot(2,2,3); histogram(sum(diffs_fg(1:4,:)),'Normalization','pdf')

figure;
subplot(2,2,1); histogram(sum(abs(diffs_sg)),'Normalization','pdf')
Y = sqrt(diag(diffs_sg'*diffs_sg));
subplot(2,2,2); histogram(Y,'Normalization','pdf')
subplot(2,2,3); histogram(sum(diffs_sg(1:4,:)),'Normalization','pdf')
% longConv = extremeG(~cellfun(@isempty,extremeG));
% lc = length(longConv);
% 
% svds = zeros(lc,K);
% for i = 1:lc
% tmp_G = longConv{i};
% tmp_div = pStar*tmp_G;
% Z = pStar'.*tmp_G./tmp_div;
% svds(i,:) = svd(Z);
% end