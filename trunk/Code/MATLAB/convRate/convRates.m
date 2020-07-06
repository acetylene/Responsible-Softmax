%TODO: make nice latex versions of these graphs!
clear;clc;
%rng(10072019);
%rng(91376491);
rng('default')
%rng(123545)
K = 5;%randi(16)+4;%number of classes
N_min = 1000;%Total sample size minimum
N_max = 500000;
%% GMM for sampling
mu = randi(20,K,1); %means
sigma = rand(1,1,K)/2+1; %variances
pStar = rand(1,K);
pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm
% figure
numiter = 50;
h = waitbar(0/numiter,sprintf('On iteration %d of %d',0,numiter));
for J=numiter:-1:1
    waitbar((numiter-J+1)/numiter,h,sprintf('On iteration %d of %d',(numiter-J+1),numiter))
    N = randi(N_max-N_min)+N_min;
    [X,T] = random(gm,N);%total sample
    
    F=zeros(K,N);
    for i=K:-1:1
        F(i,:)=normpdf(X,mu(i),sigma(1,1,i));
    end
    p_0 = ones(K,1)./K;
    [pHat,fullOrbit] = stablepoint(F,p_0,8,'diff',true);
    
    % For verification that the finite iterated method does the same as the
    % 'infinitely' iterated method.
   % [finitePi,orbitFinite] = iteratedSimplexMap(F,p_0,900);     
    M = size(fullOrbit,2);
    distancesHilb = zeros(1,M);
    distancesEuc = zeros(1,M);
    errorsHilb = zeros(1,M-1);
    errorsEuc = zeros(1,M-1);
    for n=(M-1):-1:1
        distancesHilb(n) = hilbertDistSimplex(fullOrbit(:,n),fullOrbit(:,n+1),K);
        errorsHilb(n)= hilbertDistSimplex(fullOrbit(:,n),fullOrbit(:,M),K);
    end
    
    for n=(M-1):-1:1
        distancesEuc(n) = norm(fullOrbit(:,n)-fullOrbit(:,n+1));
        errorsEuc(n) = norm(fullOrbit(:,n)-fullOrbit(:,M));
    end
    
    
    % use the first 40% for 'break in' and do not fit to the regression
    L = floor(M*.4);
%     hold on
%     plot(log(distancesEuc(L:M-1)),'-')
%     plot(log(distancesHilb(L:M-1)),'--')
%     hold off
%     pause(.05)
%     modelstr = 'y~b1+b2/x';
%     opts = statset('MaxIter',M);
%     b0 = [-1,-1];
    expMdlEuc = fitlm((1:(M-L))',log(distancesEuc(L:end-1))',...
          'RobustOpts','on');
%         modelstr,b0,'Options',opts);
    expMdlHilb = fitlm((1:(M-L))',log(distancesHilb(L:end-1))',...
          'RobustOpts','on');  
%         modelstr,b0,'Options',opts);
    rates.hilbDists{J} = distancesHilb;
    rates.hilbErr{J} = errorsHilb;
    
    rates.eucDists{J} = distancesEuc;
    rates.eucErrs{J} = errorsEuc;
    
    rates.lambdaE(J) = expMdlEuc.Coefficients.Estimate(2);
    rates.lambdaH(J) = expMdlHilb.Coefficients.Estimate(2);
    
    rates.lambdaF(:,J) = svd(F);
    rates.sz(J) = M;
    rates.samplesz(J) = N;
end
close(h);
euc = rates.lambdaE(:);
hilb = rates.lambdaH(:);
maxsvdF = 1./rates.lambdaF(1,:);
sizes = rates.samplesz(:);

figure
scatter(sizes,euc+maxsvdF',20,'b','o')
hold on
scatter(sizes,hilb+maxsvdF',20,'r','x')
hold off
title('$\sigma_n^{-1} - b_n$ vs. sample size $N_n$','Interpreter','latex')

figure
hold on
for ii=1:ceil(numiter/10)
plot(1:rates.sz(ii),log(rates.eucDists{ii}(:)));
end
title({'$\log(a_n)$ vs. $n$', 'Euclidean distance'},'Interpreter','latex')

figure
hold on
for ii=1:ceil(numiter/10)
plot(1:rates.sz(ii)-1,log(rates.eucErrs{ii}(:)));
end
title({'$\log(\epsilon_n)$ vs. $n$', 'Euclidean distance'},'Interpreter','latex')