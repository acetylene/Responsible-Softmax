%% Establish Basic Parameters
K=3; %If you change K, the evaluation of ell later will not work
N=100;

p_0 = ones(K,1)./K; %initial distribution
alpha = 1.001;
precision = 12; %should not be greater than 12

%% Calculation of Fixed point for a specific example
%The example f below should work for any K
F = [alpha*ones(1,N);[zeros(K-1,1),eye(K-1),zeros(K-1,N-K)]];
[pHatF,orbitF] = stablepoint(F,p_0,precision,'diff',true);

%small peturbation of F gives the same final answer
G = F + abs(randn(K,N)./100000);
[pHatG,orbitG] = stablepoint(G,p_0,precision,'diff',true);

ell = @(X,y) log(prod(y'*X,2))./N;

%Showing that ell increases on the orbit (-ell is Lyapunov sanity check)
ell(F,orbitF)
ell(G,orbitG)

%% Evaluate ell(G,p) on the 3 simplex. Requires K==3.
%larger resolution makes the script run faster. 
%Should be greater than about 10^-4 
resolution = .001; 
% Set up v as a 'mesh' of the 2D simplex in R^3
v=zeros(size(F));
k=1;
for i=1:-resolution:0
for j=1-i:-resolution:0
v(:,k)=[i;j;abs(1-i-j)];
k=k+1;
end
end

% Evaluate ell(G,*) for each * in the mesh v
L = zeros(1,size(v,2));

for i=1:size(v,2)
L(i) = ell(G,v(:,i));
end

% Find max of ell on the mesh
[loss,idx] = max(L);

% Confirm
norm(v(:,idx)-pHatF)
norm(v(:,idx)-pHatG)


