%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% M-file : gauss_mixture.m
%% Author: Marek Rychlik (8-22-2018)
%% 
%% This script draws the graph of G(w) for a random
%% sample.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=3;					% Number of samples
D=2;					% Dimension
sigma=1;				% Sigma
mu=1;					% Mean of normal
alpha=0.1;				% Damping
distribution='normal';			% Name of prob. distribution, see

X0 = random(distribution, mu, sigma, [N,D]);
T0 = zeros(N,1);

X1 = random(distribution, -mu, sigma, [N,D]);
T1 = ones(N,1);

X = [X0;X1];				% Mixture
T = [T0;T1];				% Targets

% scatter(X(:,1),X(:,2));		% Scatter plot samples
% pause();

[U,V] = meshgrid(-10:.2:10,-10:.2:10);

U1=U(:);				% Linearize grid X
V1=V(:);				% Linearize grid Y
W = [U1,V1];				% Create weights

Y = sigmoid(X * W');			% Compute activations
Z = -sum(T .* log(Y) + (1-T) .* log(1-Y));	% Compute M(W)
Z = Z';
Z = Z +  alpha * sum(W.^2,2);

Z=reshape(Z,size(U));			% Prepare for the mesh
mesh(U,V,Z);				% Plot the surface
%contour(U,V,Z);				% Plot the level curves

function y = sigmoid(x)
y = 1./(1+exp(-x));
end
