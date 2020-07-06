function [handles] = stablepointExplorationGMM(gm, numsamples, seed, mean_on, coords_on, ev_on, hist_on)

%% Setup of various functions to be used
G = @(F,p) F./(p'*F);
H_ell = @(F,p,N) -1/N.*G(F,p)*G(F,p)';

options = {12, 'diff', true};
stabPt = @(F,p) stablepoint(F,p,options{:});

%% Setup GM model to sample from
rng(seed)
%sig(1,1,:) = [3;.2];
%gm = gmdistribution([0;2.5],sig,[.8,.2]);
% numsamples = 5;
[X,T] = random(gm,numsamples);
p_samp = sum(T==1)/numsamples;
sampRats = [p_samp;1-p_samp];

%% F is a 1 parameter family of matrices.
mu = gm.mu;
sigma = sqrt(squeeze(gm.Sigma));
if mean_on
    % X - sample data
    % x - proposed mean
    F = @(X,x) [normpdf(X,mu(1),sigma(1)),normpdf(X,x,sigma(2))];%unkown mean
else
    F = @(X,x) [normpdf(X,mu(1),sigma(1)),normpdf(X,mu(2),abs(x+.00001))];%unkown var.
end
%% Make sure starting point is not precisely at what is most uninformative
tol = .001;
del = @(K) rand(K,1)*tol;
p = @(K) ones(K,1)./K+del(K);

ell = @(X,y,N) log(prod(y'*X,2))./N;

[K,N] = size(F(X,0.1)');

%% Set bound for parameters
startval = -1;
endval = 4;
res = .01;
numel = ceil((endval-startval)/res)+1;
stabptsF = zeros(K,numel);
hessiansF = zeros(K,K,numel);
eigvalsF = zeros(K,numel);
ellevalsF = zeros(2,numel);

%% Calculate piHat_x for x in the given range

for x = 1:numel
    evPt = res*(x-1)+startval;
    p_0 = p(K);
    p_0 = p_0./sum(p_0);
    stabptsF(:,x) = stabPt(F(X,evPt)',p_0);
    if ev_on
        hessiansF(:,:,x) = H_ell(F(X,evPt)',stabptsF(:,x),N);
        eigvalsF(:,x) = eigs(hessiansF(:,:,x));
        ellevalsF(:,x) = [ell(F(X,evPt)',p(K),N);ell(F(X,evPt)',stabptsF(:,x),N)];
    end
end
%% Plot the given values
labels{K} = '';
linestyles = [{'-.'},{'-'},{'--'}];
L = length(linestyles);

if coords_on
    h(1) = figure;
    %h(1) = gcf;
    hold on
    for k = 1:K
        labels{k} = strcat('\pi_', sprintf('%d',k));
        plot(linspace(startval,endval,numel),stabptsF(k,:),linestyles{mod(k,L)+1})
    end
    legend(labels{:},'Location','eastoutside')
    if mean_on
        title({'$\hat{\pi}$ coordinate values v. $\mu_2$';...
               sprintf('$N = %d$, $\\pi^* = (%0.2f,%0.2f)$',[N,sampRats'])},...
               'Interpreter' ,'latex')
        xlabel('value of $\mu_2$','Interpreter' ,'latex');
    else
        title({'$\hat{\pi}\text{ coordinate values v. }\sigma_2$';
               sprintf('$N = %d \\text{, }\\hat{\\pi} = (%0.2f,%0.2f)$',[N,sampRats'])},...
               'Interpreter' ,'latex')
        xlabel('value of \sigma_2');
    end
    
    axis([startval endval 0 1]);
end

if ev_on
    h(2) = figure;
    hold on
    for k = 1:K
        labels{k} = strcat('ev_', sprintf('%d',k));
        plot(linspace(startval,endval,numel),eigvalsF(k,:),linestyles{mod(k,L)+1})
    end
    legend(labels{:},'Location','eastoutside')
    title(sprintf('Eigenvalues of \nabla^2 v. X, ratio: (%0.1f,%0.1f)',sampRats));
    xlabel('value of x in F_x');
end

if hist_on
    h(3) = figure;
    hold on
    
    if (numsamples<=10)
        scatter(X,ones(size(X)),'x');
    else
        histogram(X);
    end
end

handles = h;
% k= 1;
% for x = .5:.25:2.5
% labels{k} = strcat('x = ', sprintf('%0.1f',x));
% n=1;
% ell_x = zeros(1,length(0:.01:1));
% for j = 1:-.01:0
% ell_x(n) = ell(F(X,x)',[j;1-j],N);
% n=n+1;
% end
% plot(1:-.01:0,ell_x,linestyles{mod(k,L)+1});
% k=k+1;
% end
% legend(labels{:},'Location','eastoutside')
% title('ell_x for varying x');
% xlabel('\pi_1')
% ylabel('ell_x(\pi_1,1-\pi_1)');
