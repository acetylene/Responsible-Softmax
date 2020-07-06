%% Basic Parameters
clear;clc;
rng('default')
K = 5;%randi(16)+4;%number of classes
N = 10000;%Total sample size
D = 2;
%% GMM for sampling
mu = [linspace(-K,K,K)*3.5;randi(4,1,K)-2]'; %means

for k=K:-1:1
    x = rand();
    if mod(k,2) == 0
        A = [0,-1;1,0];
        evals = [5*x,x];
    else
        %         tmp = sprand(D,D,density,rc);
        A = [1,-1;1,1]./sqrt(2);%full(tmp);
        evals = [19*x,x];
    end
    Sig = diag(evals);
    sigma(:,:,k) = A*Sig*A' ; %variances
end
%adjust for specific example
mu(2,1) = -9;
mu(4,1) = 9;
sigma(:,:,4) = sigma(:,:,2);
sigma(:,:,5) = sigma(:,:,1);
idx = [1,3,5];%unique(randi(K,1,K-1));
pStar = rand(1,K);
pStar(idx) = pStar(idx)+10;
pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm

X=random(gm,N);
%% Inline functions to describe derivatives
for i=K:-1:1
    f{i} = @(x) mvnpdf(x,mu(i,:),squeeze(sigma(:,:,i)));
end
options = {6, 'diff', false};
stabPt = @(F,p) stablepoint(F,p,options{:});
G = @(F,p) F./(p'*F);
H_ell = @(F,p,N) -1/N.*G(F,p)*G(F,p)';
F = @(X) [f{1}(X),f{2}(X),f{3}(X),f{4}(X),f{5}(X)];
d=@(F,p,N) 1/N*(G(F,p))*ones(N,1);
DR = @(F,p,N) diag(p)*H_ell(F,p,N)+diag(d(F,p,N));
%% Small meshgrid to evaluate on.
resolution =0.1;
k=1;
for ii1 = 0.1:resolution:.9
    for ii2 = 0.1:resolution:(.9-ii1)
        for ii3 = 0.1:resolution:(.9-ii1-ii2)
            for ii4 = 0.1:resolution:(.9-ii1-ii2-ii3)
                w(:,k) = [ii1;ii2;ii3;ii4;abs(1-ii1-ii2-ii3-ii4)];
                %sprintf('[%5.2f, %5.2f, %5.2f, %5.2f, %5.2f]',w(:,k))
                k=k+1;
            end
        end
    end
end

for i = size(w,2):-1:1
    H = H_ell(F(X)',w(:,i),N);
    PH = diag(w(:,i))*H;
    dR = DR(F(X)',w(:,i),N);
    %capture eigenvalues of H = hessian of ell
    evOnSimp{1}(:,i) = eigs(H);
    %capture eigenvalues of PH = DR+d
    evOnSimp{2}(:,i) = eigs(PH);
    %capture eigenvalues of DR = Frechet derivative of R
    evOnSimp{3}(:,i) = eigs(dR);
end
plotson = false;%true;
len = length(evOnSimp);

if plotson
    for ii=len:-1:1
        eVals = evOnSimp{len-ii+1};
        handles{ii} = figure;
        plotmatrix(eVals');
        handles{ii+len}=figure;
        hold on
        for jj = 1:5
            plot(1:size(w,2),eVals(jj,:));
        end
        hold off
    end
end

%% Only run below overnight! (~7hour run time)
willingtowait = false;
if willingtowait
    resolution =0.01;
    n=1;
    for i1 = 0.01:resolution:.99
        for i2 = 0.01:resolution:.99-i1
            for i3 = 0.01:resolution:.99-i1-i2
                for i4 = 0.01:resolution:.99-i1-i2-i3
                    v(:,n) = [i1;i2;i3;i4;abs(1-i1-i2-i3-i4)];
                    n=n+1;
                end
            end
        end
    end
    
    h = waitbar(0/size(v,2),sprintf('On point %d of %d',0,size(v,2)));
    for i = size(v,2):-1:1
        H = H_ell(F(X)',v(:,i),N);
        PH = diag(v(:,i))*H;
        dR = DR(F(X)',v(:,i),N);
        evOnSimpBig{1}(:,i) = eigs(H);
        evOnSimpBig{2}(:,i) = eigs(PH);
        evOnSimpBig{3}(:,i) = eigs(dR);
        waitbar((size(v,2)-i+1)/size(v,2),h,sprintf('On point %d of %d',size(v,2)-i+1,size(v,2)));
    end
end