%% Setup of various functions to be used
G = @(F,p) F./(p'*F);
H_ell = @(F,p,N) -1/N.*G(F,p)*G(F,p)';

options = {12, 'diff', true};
stabPt = @(F,p) stablepoint(F,p,options{:});

rng('default')
%A = rand(2)+1.5;
K = 2;
A = @(x,a) [eye(K);zeros(1,K-2), x*a,x*(1-a)];
U = rand(1,6);
V = 2*U;

% F is a 1 parameter family of matrices.
F =@(x) [[1,x;2,1;0,0],[U,V;V,U;ones(1,12)]];
%% Other options for F
%Since F is what is being exlpored we need to try serveral options. Maybe a
%GUE, or something from a 2 class GMM where means approach each other?
%S = rand(2,10);
% A(x,.4)*S;
%diag([1,(2*x-.5)^2])+[0, 1;1, 0];%
%add below to the end of F (for K=2) to 'add more samples'
%,repmat([2,1;1,2],1,1)];
%[1,x,x,.001,.001;1-x,1,1-x,.001,.001;.001,.001,1-x,1,x;.001,.001,x,1-x,1];

%% Make sure starting point is not precisely at what is most uninformative
tol = .001;
sig = @(K) rand(K,1)*tol;
p_0 = @(K) ones(K,1)./K+sig(K);

ell = @(X,y,N) log(prod(y'*X,2))./N;

[K,N] = size(F(0.1));

%% Set bound for parameters
endval = 8;
startval = 0;
res = .01;
numel = ceil((endval-startval)/res)+1;
stabptsF = zeros(K,numel);
hessiansF = zeros(K,K,numel);
eigvalsF = zeros(K,numel);
ellevalsF = zeros(2,numel);

%% Calculate piHat_x for x in the given range
for x = 1:numel
   evPt = res*(x-1)+startval;
   p_0 = p_0(K);
   p_0 = p_0./sum(p_0);
   stabptsF(:,x) = stabPt(F(evPt),p_0); 
   hessiansF(:,:,x) = H_ell(F(evPt),stabptsF(:,x),N);
   eigvalsF(:,x) = eigs(hessiansF(:,:,x));
   ellevalsF(:,x) = [ell(F(evPt),p_0(K),N);ell(F(evPt),stabptsF(:,x),N)];
end

%% Plot the given values
labels{K} = '';
linestyles = [{'-.'},{'-'},{'--'}];
L = length(linestyles);
h(1) = figure;
hold on
for k = 1:K
    labels{k} = strcat('\pi_', sprintf('%d',k));
    plot(linspace(startval,endval,numel),stabptsF(k,:),linestyles{mod(k,L)+1})
end
legend(labels{:},'Location','eastoutside')
title('$\hat{\pi}_\alpha$ coordinate values v. $\alpha$','interpreter','latex');
xlabel('value of $\alpha$ in $F_\alpha$','interpreter','latex');

h(2) = figure;
hold on
for k = 1:K
    labels{k} = strcat('\lambda', sprintf('%d',k));
    plot(linspace(startval,endval,numel),eigvalsF(k,:),linestyles{mod(k,L)+1})
end
legend(labels{:},'Location','eastoutside')
title('Eigenvalues of $\nabla^2$ v. $\alpha$','interpreter','latex');
xlabel('value of $\alpha$ in $F_\alpha$','interpreter','latex');

h(3) = figure;
hold on
k= 1;
% labels = {};
for x = .5:.25:2.5
labels{k} = strcat('$\alpha = $', sprintf('%0.2f',x));
n=1;
% ell_x = zeros(1,length(0:.01:1));
for j = 1:-.001:0
    ell_x(:,n) = [j,ell(F(x),[j;1-j],N)];
    n=n+1;
end
[~,argmax] = max(ell_x(2,:));
plot(ell_x(1,:),ell_x(2,:),...
    strcat(linestyles{mod(k,L)+1},'o'),...
    'MarkerIndices',argmax,...
    'MarkerFaceColor','k');
k=k+1;
end
legend(labels{:},'Location','eastoutside','interpreter','latex')
title('$\ell_\alpha$ for varying $\alpha$','interpreter','latex');
xlabel('\pi_1')
ylabel('$\ell_\alpha(\pi_1,1-\pi_1)$','interpreter','latex')
% zlabel('$\ell_\alpha(\pi_1,\pi_2,1-\pi_1-\pi_2)$','interpreter','latex');
% ax = gca;
% ax.View = [50.2, 18];
